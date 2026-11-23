"""Targeted 6-case TRAIN-only ablation for the Joint4 native/GNN shortlist patch.

Run from the DRL repository root after applying joint4_native_gnn_diversity.patch.
Uses existing Joint4 weights and B32; does not retrain, change q, modify baselines,
or access validation/test. Reuses the existing Joint4 results CSV with a distinct
phase so completed historical experiments are not silently reused or overwritten.
"""
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
import torch
import run_gnn_weld_layout as e

PHASE = "screen_ablation_20261123"
SEED = e.SEED


def main():
    e.configure_cpu()
    saved = torch.load(e.JOINT4_MODEL, map_location="cpu", weights_only=True)
    if not saved.get("training_complete"):
        raise ValueError("A trained Joint4 model is required")
    model = e.joint4_model(saved["state_dict"]).eval()
    model.requires_grad_(False)
    layouts = dict(e.ppo_layouts("train"))
    for name in e.JOINT3B_TRAIN_CASES:
        e.joint3_case(
            name, layouts[name], SEED, model, e.joint3_references(),
            budget=e.JOINT3_BUDGET, variant="joint4", sink="joint4",
            phase=PHASE, limit_mode="relative", write=True,
        )
    rows = e.read_rows(e.JOINT4_RESULTS)
    old = {(r["layout_id"], int(r["seed"])): r for r in rows
           if r.get("phase") == "train" and r.get("stage") == "final"
           and r.get("variant") == "joint4"}
    new = {(r["layout_id"], int(r["seed"])): r for r in rows
           if r.get("phase") == PHASE and r.get("stage") == "final"
           and r.get("variant") == "joint4"}
    deltas = []
    print("layout,status_old,status_new,Cmax_old_s,Cmax_new_s,delta_pct,"
          "Pmax_old_s,Pmax_new_s,spread_old_s,spread_new_s,"
          "MOVE_old_s,MOVE_new_s,WAIT_old_s,WAIT_new_s,B32_old,B32_new")
    for name in e.JOINT3B_TRAIN_CASES:
        key = (name, SEED)
        a, b = old.get(key), new.get(key)
        if not a or not b:
            print(f"{name},MISSING_RECORD")
            continue
        if a["status"] == b["status"] == "FEASIBLE":
            old_c, new_c = float(a["cmax_s"]), float(b["cmax_s"])
            delta = 100 * (new_c - old_c) / old_c
            deltas.append(delta)
            print(",".join(map(str, (name, a["status"], b["status"], old_c, new_c, delta,
                a["max_processing_s"], b["max_processing_s"],
                a["processing_spread_s"], b["processing_spread_s"],
                a["move_s"], b["move_s"], a["wait_s"], b["wait_s"],
                a["b32_calls"], b["b32_calls"]))))
        else:
            print(f"{name},{a['status']},{b['status']},NO_PAIRED_FEASIBLE_CMAX")
    print("PAIRED", len(deltas), "MEDIAN_DELTA_PCT", median(deltas) if deltas else None,
          "WINS", sum(x < -1e-9 for x in deltas),
          "LOSSES", sum(x > 1e-9 for x in deltas))


if __name__ == "__main__":
    main()
