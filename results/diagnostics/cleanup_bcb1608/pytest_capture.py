"""Diagnostic evidence collection only; does not change test outcomes."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import pytest

_items = {}
_reports = {}
_deselected = []
_collection_errors = []
_started = time.monotonic()


def _stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def _parameter(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return {"type": type(value).__qualname__, "repr": repr(value)}


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        callspec = getattr(item, "callspec", None)
        _items[item.nodeid] = {
            "nodeid": item.nodeid,
            "test_file": str(item.path),
            "test_function": getattr(item, "originalname", None) or item.name,
            "test_name_with_parameters": item.name,
            "parameters": {} if callspec is None else {
                key: _parameter(value) for key, value in callspec.params.items()
            },
            "location": {"file": str(item.location[0]), "line": item.location[1] + 1},
        }
    if os.environ.get("MRTA_DIAG_FILTER") == "common":
        data = json.loads(Path(os.environ["MRTA_DIAG_COMPARISON"]).read_text(encoding="utf-8"))
        allowed = set(data["common_nodeids"])
        if not allowed:
            raise RuntimeError("No shared non-RL tests were identified")
        chosen = [item for item in items if item.nodeid in allowed]
        excluded = [item for item in items if item.nodeid not in allowed]
        items[:] = chosen
        _deselected.extend(item.nodeid for item in excluded)
        config.hook.pytest_deselected(items=excluded)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append({"nodeid": report.nodeid, "traceback": str(report.longrepr)})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    record = _reports.setdefault(item.nodeid, {"nodeid": item.nodeid, "phases": {}})
    phase = {"outcome": report.outcome, "duration_s": report.duration}
    if report.failed:
        phase["console_traceback"] = report.longreprtext
        if call.excinfo is not None:
            info = call.excinfo
            phase["exception_type"] = info.type.__module__ + "." + info.type.__qualname__
            phase["exception_message"] = str(info.value)
            phase["python_traceback"] = "".join(traceback.format_exception(info.type, info.value, info.tb))
            try:
                phase["full_traceback"] = str(info.getrepr(style="long", showlocals=False,
                                                          tbfilter=False, chain=True))
            except Exception as error:
                phase["full_traceback"] = phase["python_traceback"]
                phase["formatting_note"] = repr(error)
            frames = []
            root = Path(item.config.rootpath).resolve()
            for entry in info.traceback:
                source_path = Path(str(entry.path)).resolve()
                try:
                    statement = str(entry.statement)
                except Exception:
                    statement = ""
                frames.append({
                    "file": str(source_path), "line": entry.lineno + 1,
                    "function": entry.name, "statement": statement,
                    "project_frame": source_path.is_relative_to(root),
                })
            phase["frames"] = frames
            phase["exception_origin"] = frames[-1] if frames else None
            project_frames = [frame for frame in frames if frame["project_frame"]]
            phase["first_project_frame"] = project_frames[0] if project_frames else None
            phase["failure_assertion_or_statement"] = frames[-1]["statement"] if frames else ""
        phase["captured_sections"] = [{"name": name, "text": text} for name, text in report.sections]
    record["phases"][report.when] = phase


def pytest_sessionfinish(session, exitstatus):
    root = Path(session.config.rootpath).resolve()
    versions = {}
    for name in ("pytest", "numpy", "numba", "llvmlite", "torch", "openpyxl", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    records = []
    for nodeid, report in _reports.items():
        entry = dict(_items.get(nodeid, {"nodeid": nodeid}))
        entry.update(report)
        phases = list(report["phases"].values())
        entry["outcome"] = ("failed" if any(p["outcome"] == "failed" for p in phases)
                            else "skipped" if any(p["outcome"] == "skipped" for p in phases)
                            else "passed")
        records.append(entry)
    payload = {
        "label": os.environ.get("MRTA_DIAG_LABEL"), "finished_at": _stamp(),
        "root": str(root), "executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "versions": versions,
        "argv": sys.argv, "elapsed_s": time.monotonic() - _started,
        "exitstatus": int(exitstatus), "testscollected": session.testscollected,
        "environment": {key: os.environ.get(key) for key in (
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
            "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "TEMP", "TMP", "MRTA_PLATFORM_ROOT", "MRTA_PPO_ROOT")},
        "external_geometry_root": str((root.parent / "ppo").resolve()),
        "collection": list(_items.values()), "deselected": _deselected,
        "collection_errors": _collection_errors, "tests": records,
        "counts": {outcome: sum(r["outcome"] == outcome for r in records)
                   for outcome in ("passed", "failed", "skipped")},
    }
    destination = Path(os.environ["MRTA_DIAG_OUTPUT"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(content.encode("utf-8")) >= 100000000:
        raise RuntimeError("Diagnostic JSON exceeds the requested file-size limit")
    destination.write_text(content + "\n", encoding="utf-8")
