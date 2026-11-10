#!/usr/bin/env python3
"""用排样 Excel 在 20×12 m 平台上做二维随机排样，输出标注 PNG。

从型号库随机抽取，可重复放置同一型号，直到剩余空间放不下任何型号。
底板与焊缝共用平面旋转和平移，默认件间净距为 0.3 m，可用 --spacing 指定。
每个净距多次试排，选取矩形占用面积最大的结果。
焊缝按板面 U/V 投影。投影长度不超过 0.001 mm 的线段跳过，连续的近切向
线段沿原顶点合成一条折线焊缝。
用 --split train 或 --split test 导出 JSONL。导出时保留 165-SR2A 的第 9 号
直线焊缝，删去同一母焊缝两端的弧线。其他焊缝的连续短弧按 0.5 m 规则处理。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations, product
from math import cos, hypot, isfinite, radians
from pathlib import Path
from random import Random

import numpy as np

from reconstruct_assembly_structure import INK, MUTED, WELD, Weld
from reconstruct_layout_geometry import LayoutPart, load_layout_parts

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator


PROJECT_DIR = Path(__file__).resolve().parent
PLATFORM_WIDTH = 20.0
PLATFORM_HEIGHT = 12.0
SPACING = 0.3
PACKING_TRIALS = 1024
TOLERANCE_MM = 0.001
TOLERANCE_M = 1e-10
MIN_CONTINUATION_COS = cos(radians(30))
MIN_CURVED_LENGTH_MM = 500.0
FIT_TOLERANCE_MM = 30.0
FIT_SAMPLE_MM = 0.5
DATASET_SEEDS = {"train": range(2026111001, 2026111013), "test": range(2026111013, 2026111017)}
PLATE_FILL = "#DBEDE2"
PLATE_EDGE = "#49775E"
BOX_EDGE = "#506B8B"
Box = tuple[float, float, float, float]


@dataclass
class WeldPath:
    numbers: tuple[int, ...]
    uv_mm: np.ndarray
    length_mm: float


@dataclass
class Placement:
    part: LayoutPart
    x: float
    y: float
    quarter_turns: int

    @property
    def bounds(self) -> Box:
        width, height = self.part.width / 1000, self.part.height / 1000
        if self.quarter_turns % 2:
            width, height = height, width
        return self.x, self.y, self.x + width, self.y + height

    def project(self, uv_mm: np.ndarray) -> np.ndarray:
        u, v = uv_mm.T / 1000
        width, height = self.part.width / 1000, self.part.height / 1000
        x, y = ((u, v), (height - v, u), (width - u, height - v), (v, width - u))[self.quarter_turns]
        return np.column_stack((x + self.x, y + self.y))


def planar_weld_paths(welds: list[Weld]) -> list[WeldPath]:
    segments = [w for w in welds if np.linalg.norm(w.points[1, :2] - w.points[0, :2]) > TOLERANCE_MM]
    if not segments:
        return []

    points = np.array([w.points for w in segments]).reshape(-1, 3)
    segment_ids = np.arange(len(points)) // 2
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    adjacent = (distances <= TOLERANCE_MM) & (segment_ids[:, None] != segment_ids[None, :])
    connections = {}
    for i, row in enumerate(adjacent):
        neighbors = np.flatnonzero(row)
        if len(neighbors) != 1:
            continue
        j = int(neighbors[0])
        if j < i or adjacent[j].sum() != 1:
            continue
        # 同一点上的两条线需要沿同一方向延续。分叉和急转角各自保留。
        a, b = points[i ^ 1] - points[i], points[j ^ 1] - points[j]
        if a @ b <= -MIN_CONTINUATION_COS * np.linalg.norm(a) * np.linalg.norm(b):
            connections[i], connections[j] = j, i

    starts = [i for i in range(len(points)) if i not in connections]
    starts.extend(range(0, len(points), 2))  # 也能从任意一段开始遍历闭合焊缝。
    remaining = set(range(len(segments)))
    paths = []
    for start in starts:
        if start // 2 not in remaining:
            continue
        current = start
        vertices = [points[current, :2]]
        numbers = []
        length = 0.0
        while current is not None and current // 2 in remaining:
            index = current // 2
            remaining.remove(index)
            numbers.append(segments[index].number)
            length += segments[index].length_mm
            end = current ^ 1
            vertices.append(points[end, :2])
            current = connections.get(end)
        paths.append(WeldPath(tuple(numbers), np.array(vertices), length))
    return paths


def point_to_path(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    edges = np.diff(path, axis=0)
    offsets = points[:, None, :] - path[:-1]
    squared = np.sum(edges**2, axis=1)
    t = np.clip(np.sum(offsets * edges, axis=2) / np.where(squared > 0, squared, 1), 0, 1)
    return np.linalg.norm(offsets - t[..., None] * edges, axis=2).min(axis=1)


def sample_polyline(path: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.linspace(a, b, int(np.ceil(np.linalg.norm(b - a) / FIT_SAMPLE_MM)) + 1)
        for a, b in zip(path, path[1:])
    ])


def fit_retained_curve(raw: np.ndarray, part: LayoutPart) -> np.ndarray:
    """Try supporting lines on the plate side of a concave, U-monotone curve."""
    reverse = raw[0, 0] > raw[-1, 0]
    points = raw[::-1] if reverse else raw
    delta = np.diff(points, axis=0)
    if len(delta) < 3 or (delta[:, 0] <= 0).any():
        return raw
    slopes = delta[:, 1] / delta[:, 0]
    if (np.diff(slopes) >= 0).any():
        return raw
    intercepts = points[:-1, 1] - slopes * points[:-1, 0]
    outlines = [np.vstack((outline, outline[0])) for outline, _, _ in part.slabs]
    polygons = [PolygonPath(outline, closed=True) for outline in outlines]

    def outside_distance(samples: np.ndarray) -> float:
        inside = np.logical_or.reduce([polygon.contains_points(samples) for polygon in polygons])
        if inside.all():
            return 0.0
        outside = samples[~inside]
        return float(np.minimum.reduce([point_to_path(outside, outline) for outline in outlines]).max())

    source = sample_polyline(points)
    original_outside = outside_distance(source)
    support = list(range(len(delta)))
    fitted = raw
    for edge in range(1, len(delta) - 1):
        remaining = [index for index in support if index != edge]
        a, b = np.array(remaining[:-1]), np.array(remaining[1:])
        u = (intercepts[b] - intercepts[a]) / (slopes[a] - slopes[b])
        candidate = np.vstack((points[0], np.column_stack((u, slopes[a] * u + intercepts[a])), points[-1]))
        if (np.diff(candidate[:, 0]) <= 0).any():
            continue
        sampled = sample_polyline(candidate)
        error = max(point_to_path(source, candidate).max(), point_to_path(sampled, points).max())
        # Half the sample spacing bounds the error between samples.
        if error + FIT_SAMPLE_MM / 2 <= FIT_TOLERANCE_MM and outside_distance(sampled) <= original_outside + 0.01:
            support = remaining
            fitted = candidate[::-1] if reverse else candidate
    return fitted


def dataset_weld_paths(part: LayoutPart) -> list[list[WeldPath]]:
    """Return DRL straight tasks grouped by their source weld path."""
    welds = part.welds
    if part.ident == "165-SR2A":
        # 原焊缝 4–8 和 10–13 是第 9 号直线两侧的弧段。
        welds = [weld for weld in welds if weld.number == 9 or not 4 <= weld.number <= 13]
    paths = []
    for path in planar_weld_paths(welds):
        raw, numbers = path.uv_mm, path.numbers
        if len(numbers) == 1:
            paths.append([path])
            continue

        chord = raw[-1] - raw[0]
        if (np.diff(raw, axis=0) @ chord > 0).all() and point_to_path(raw, raw[[0, -1]]).max() <= TOLERANCE_MM:
            paths.append([WeldPath(numbers, raw[[0, -1]], float(np.linalg.norm(chord)))])
            continue

        lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        if lengths.sum() < MIN_CURVED_LENGTH_MM:
            continue
        middle = int(lengths.argmax())
        if lengths[middle] >= MIN_CURVED_LENGTH_MM and np.linalg.norm(chord) > TOLERANCE_MM:
            start = middle if lengths[:middle].sum() < MIN_CURVED_LENGTH_MM else 0
            end = middle + 1 if lengths[middle + 1:].sum() < MIN_CURVED_LENGTH_MM else len(lengths)
            raw, numbers = raw[start:end + 1], numbers[start:end]

        fitted = fit_retained_curve(raw, part)
        starts = np.minimum(raw[:-1, 0], raw[1:, 0])
        ends = np.maximum(raw[:-1, 0], raw[1:, 0])
        segments = []
        for index, (a, b) in enumerate(zip(fitted, fitted[1:])):
            if fitted is raw:
                source = (numbers[index],)
            else:
                low, high = sorted((a[0], b[0]))
                overlap = np.minimum(ends, high) - np.maximum(starts, low)
                source = tuple(number for number, span in zip(numbers, overlap) if span > 1e-6)
            segments.append(WeldPath(source, np.array([a, b]), float(np.linalg.norm(b - a))))
        paths.append(segments)
    return paths


def split_free_space(free: list[Box], used: Box) -> list[Box]:
    x, y, right, top = used
    available = []
    for a, b, c, d in free:
        if right <= a + TOLERANCE_M or top <= b + TOLERANCE_M or x >= c - TOLERANCE_M or y >= d - TOLERANCE_M:
            available.append((a, b, c, d))
            continue
        if x > a + TOLERANCE_M:
            available.append((a, b, x, d))
        if right < c - TOLERANCE_M:
            available.append((right, b, c, d))
        if y > b + TOLERANCE_M:
            available.append((a, b, c, y))
        if top < d - TOLERANCE_M:
            available.append((a, top, c, d))
    unique = list(dict.fromkeys(available))
    return [r for i, r in enumerate(unique) if not any(
        j != i and a <= r[0] and b <= r[1] and c >= r[2] and d >= r[3]
        for j, (a, b, c, d) in enumerate(unique)
    )]


def choose_position(part: LayoutPart, free: list[Box], spacing: float, rng: Random,
                    strategy: int) -> tuple[Placement, Box] | None:
    width, height = part.width / 1000 + spacing, part.height / 1000 + spacing
    options = []
    for left, bottom, right, top in free:
        for turns, (w, h) in enumerate(((width, height), (height, width))):
            dx, dy = right - left - w, top - bottom - h
            if min(dx, dy) < -TOLERANCE_M:
                continue
            short, long = min(dx, dy), max(dx, dy)
            unused = (right - left) * (top - bottom) - w * h
            score = ((short, long), (unused, short), (long, short))[strategy]
            options.append((score, rng.random(), left, bottom, dx, dy, w, h, turns))
    if not options:
        return None
    _, _, x, y, dx, dy, w, h, turns = min(options)
    x += rng.choice((0, max(dx, 0)))
    y += rng.choice((0, max(dy, 0)))
    return Placement(part, x, y, turns + 2 * rng.randrange(2)), (x, y, x + w, y + h)


def pack_once(parts: list[LayoutPart], rng: Random, attempt: int, spacing: float = SPACING) -> list[Placement]:
    # 矩形的右、上各留一份净距。平台相应放宽，使留白只约束件间距离。
    free = [(0.0, 0.0, PLATFORM_WIDTH + spacing, PLATFORM_HEIGHT + spacing)]
    placed = []
    while free:
        # 每次放置后仍从整个型号库抽取。所有型号均找不到位置时才结束。
        for part in rng.sample(parts, len(parts)):
            position = choose_position(part, free, spacing, rng, attempt % 3)
            if position is not None:
                placement, occupied = position
                placed.append(placement)
                free = split_free_space(free, occupied)
                break
        else:
            break
    return placed


def occupied_area(placements: list[Placement]) -> float:
    return sum(p.part.width * p.part.height for p in placements) / 1e6


def pack_platform(parts: list[LayoutPart], seed: int, spacing: float = SPACING) -> list[Placement]:
    rng = Random(seed)
    return max((pack_once(parts, rng, trial, spacing) for trial in range(PACKING_TRIALS)), key=occupied_area)


def minimum_clearance(placements: list[Placement]) -> float:
    gaps = []
    for (ax, ay, ar, at), (bx, by, br, bt) in combinations((p.bounds for p in placements), 2):
        gaps.append(hypot(max(ax - br, bx - ar, 0), max(ay - bt, by - at, 0)))
    return min(gaps, default=float("inf"))


def dataset_record(placements: list[Placement], welds: dict[str, list[list[WeldPath]]],
                   layout_id: str, seed: int, spacing: float, workbook: Path) -> dict:
    parts, parents = [], []
    path_count = curved_count = 0
    total_length = 0.0
    for index, placement in enumerate(placements, 1):
        part_id = f"P{index:02d}"
        name = placement.part.ident
        parts.append({"part_id": part_id, "model_id": name, "rotation_deg": placement.quarter_turns * 90,
                      "bounds_xyxy_m": list(placement.bounds)})
        for number, group in enumerate(welds[name], 1):
            path_id = f"{part_id}-W{number:02d}"
            path_count += 1
            curved_count += len(group) > 1
            for segment, path in enumerate(group, 1):
                start, end = placement.project(path.uv_mm).tolist()
                total_length += hypot(end[0] - start[0], end[1] - start[1])
                parents.append({"parent_id": f"{path_id}-S{segment:02d}", "part_id": part_id,
                                "weld_path_id": path_id, "segment_index": segment,
                                "source_weld_numbers": list(path.numbers), "start": start, "end": end})

    source = workbook.resolve()
    if source.is_relative_to(PROJECT_DIR.parent):
        source = source.relative_to(PROJECT_DIR.parent)
    clearance = minimum_clearance(placements)
    return {"layout_id": layout_id, "geometry_seed": seed, "source_workbook": source.as_posix(),
            "platform_size_xy_m": [PLATFORM_WIDTH, PLATFORM_HEIGHT], "spacing_m": spacing,
            "minimum_clearance_m": clearance if isfinite(clearance) else None,
            "rectangle_utilization": occupied_area(placements) / (PLATFORM_WIDTH * PLATFORM_HEIGHT),
            "weld_path_count": path_count, "curved_path_count": curved_count, "straight_task_count": len(parents),
            "total_weld_length_xy_m": total_length, "parts": parts, "parents": parents}


def export_dataset(parts: list[LayoutPart], split: str, seeds: list[int], gaps: list[float],
                   workbook: Path, output_dir: Path) -> None:
    def held_out(name: str) -> bool:
        return name.startswith(("145-", "804-")) or name == "642-BL14A"

    library = [part for part in parts if held_out(part.ident) == (split == "test")]
    if not library:
        raise ValueError(f"{split} 没有可用的小组立型号")
    welds = {part.ident: dataset_weld_paths(part) for part in library}
    rows = []
    for index, (spacing, seed) in enumerate(product(gaps, seeds), 1):
        placements = pack_platform(library, seed, spacing)
        if not placements:
            raise ValueError(f"净距 {spacing:g} m 时，平台放不下任何型号")
        record = dataset_record(placements, welds, f"{split}_{index:02d}", seed, spacing, workbook)
        rows.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}.jsonl"
    output.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"已导出 {len(rows)} 组排样到 {output.resolve()}")


def label_position(box: Box, paths: list[np.ndarray], ident: str, points_per_meter: float):
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    vertical = width < 1.35 and height > width
    available = height if vertical else width
    size = min(9.5, ((available - 0.1) * points_per_meter - 4) / (0.6 * len(ident)))
    long = (0.6 * size * len(ident) + 4) / points_per_meter
    short = (1.15 * size + 3) / points_per_meter
    dx, dy = (short / 2, long / 2) if vertical else (long / 2, short / 2)
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    candidates = np.array(list(product(
        np.linspace(x0 + dx + 0.045, x1 - dx - 0.045, 5),
        np.linspace(y0 + dy + 0.045, y1 - dy - 0.045, 5),
    )))
    if not paths:
        return center, size, 90 if vertical else 0
    starts = np.vstack([line[:-1] for line in paths])
    vectors = np.vstack([np.diff(line, axis=0) for line in paths])
    offsets = candidates[:, None] - starts
    t = np.clip(np.sum(offsets * vectors, axis=2) / np.maximum(np.sum(vectors**2, axis=1), 1e-20), 0, 1)
    distance = np.linalg.norm(offsets - t[:, :, None] * vectors, axis=2).min(axis=1)
    scores = np.minimum(distance, 0.48) - 0.08 * np.linalg.norm(candidates - center, axis=1)
    return candidates[np.argmax(scores)], size, 90 if vertical else 0


def render_platform(placements: list[Placement], welds: dict[str, list[WeldPath]], seed: int, output: Path,
                    spacing: float = SPACING) -> None:
    available = {font.name for font in fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available), "DejaVu Sans")
    gap = minimum_clearance(placements)
    count = sum(len(welds[p.part.ident]) for p in placements)
    utilization = occupied_area(placements) / (PLATFORM_WIDTH * PLATFORM_HEIGHT)
    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(16, 10.5), facecolor="#F7FAFB")
        fig.text(0.08, 0.94, "焊接平台二维排样", fontsize=24, weight="bold", color=INK)
        fig.text(0.08, 0.903, f"随机矩形排样    ·    目标净距 {spacing:g} m    ·    随机种子 {seed}", fontsize=11, color=MUTED)
        fig.text(0.92, 0.941, f"20 × 12 m    ·    {len(placements)} 件小组立", ha="right", fontsize=16, color=INK)
        fig.text(0.92, 0.904, f"矩形面积利用率 {utilization:.1%}    ·    实测最小净距 {gap:.3f} m",
                 ha="right", fontsize=11, color=MUTED)
        fig.legend(handles=[Patch(facecolor=PLATE_FILL, edgecolor=PLATE_EDGE, label="底板外缘"),
                            Line2D([0], [0], color=BOX_EDGE, linestyle="--", label="排样矩形"),
                            Line2D([0], [0], color=WELD, linewidth=2, label="焊缝中心线")],
                   loc="center left", bbox_to_anchor=(0.074, 0.863), ncol=3, frameon=False, fontsize=10)
        ax = fig.add_axes([0.072, 0.115, 0.856, 0.719], facecolor="#FEFFFF")
        ax.set(xlim=(-0.12, PLATFORM_WIDTH + 0.12), ylim=(-0.12, PLATFORM_HEIGHT + 0.12), xlabel="平台 X (m)", ylabel="平台 Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.apply_aspect()
        points_per_meter = 72 * ax.get_window_extent().width / fig.dpi / (PLATFORM_WIDTH + 0.24)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_axisbelow(True)
        ax.grid(which="major", linewidth=0.7, color="#DCE6EA")
        ax.grid(which="minor", linewidth=0.4, color="#EEF3F5")
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(Rectangle((0, 0), PLATFORM_WIDTH, PLATFORM_HEIGHT, fill=False, edgecolor=INK, linewidth=1.35, zorder=7))

        for placement in placements:
            part = placement.part
            x0, y0, x1, y1 = placement.bounds
            for outline, _, _ in part.slabs:
                ax.add_patch(Polygon(placement.project(outline), closed=True, facecolor=PLATE_FILL,
                                     edgecolor=PLATE_EDGE, linewidth=0.75, zorder=2))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=BOX_EDGE,
                                   linestyle="--", linewidth=0.75, zorder=3))
            paths = [placement.project(path.uv_mm) for path in welds[part.ident]]
            ax.add_collection(LineCollection(paths, colors=WELD, linewidths=1.75, capstyle="butt", joinstyle="round", zorder=4))
            position, size, rotation = label_position(placement.bounds, paths, part.ident, points_per_meter)
            ax.text(*position, part.ident, ha="center", va="center", fontsize=size, weight="bold", color=INK,
                    rotation=rotation, zorder=5, bbox={"boxstyle": "round,pad=0.18", "fc": "#FFFFFF", "ec": "none", "alpha": 0.9})

        fig.text(0.08, 0.034, f"{count} 条焊缝路径", fontsize=9.5, color=MUTED)
        if any(p.part.ident == "165-SR2A" for p in placements):
            curve = next((path for path in welds["165-SR2A"] if len(path.numbers) > 1), None)
            if curve is not None:
                fig.text(0.92, 0.034, f"165-SR2A 的弧线由 {len(curve.numbers)} 段合为一条焊缝",
                         ha="right", fontsize=9.5, color=MUTED)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, nargs="?", default=PROJECT_DIR / "C_layout_geometry.xlsx")
    parser.add_argument("--seed", type=int, action="append", help="随机种子，可重复指定")
    parser.add_argument("--spacing", type=float, action="append", help=f"件间净距（m），可重复指定，默认 {SPACING:g}")
    parser.add_argument("--split", choices=tuple(DATASET_SEEDS), help="导出训练集或测试集 JSONL")
    parser.add_argument("--output-dir", type=Path, help="指定 PNG 或 JSONL 的输出目录")
    args = parser.parse_args()
    gaps = list(dict.fromkeys(args.spacing or [SPACING]))
    if any(not isfinite(spacing) or spacing <= 0 for spacing in gaps):
        parser.error("件间净距须为正数")
    try:
        parts = list(load_layout_parts(args.workbook).values())
        if not parts:
            raise ValueError("工作簿中没有小组立")
        if args.split:
            output_dir = args.output_dir or PROJECT_DIR.parent / "data" / "weld_layout"
            seeds = list(dict.fromkeys(args.seed or DATASET_SEEDS[args.split]))
            export_dataset(parts, args.split, seeds, gaps, args.workbook, output_dir)
            return
        welds = {part.ident: planar_weld_paths(part.welds) for part in parts}
        output_dir = args.output_dir or PROJECT_DIR / "platform_previews"
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = list(dict.fromkeys(args.seed or ([31] if args.spacing else [31, 2026])))
        for spacing, seed in product(gaps, seeds):
            placed = pack_platform(parts, seed, spacing)
            if not placed:
                raise ValueError(f"净距 {spacing:g} m 时，平台放不下任何型号")
            output = output_dir / f"platform_layout_gap_{spacing:g}m_seed{seed}.png"
            render_platform(placed, welds, seed, output, spacing)
            utilization = occupied_area(placed) / (PLATFORM_WIDTH * PLATFORM_HEIGHT)
            print(f"净距 {spacing:g} m，放置 {len(placed)} 件，矩形面积利用率 {utilization:.1%}，"
                  f"实测最小净距 {minimum_clearance(placed):.3f} m，输出 {output.resolve()}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"排样失败：{error}\n")


if __name__ == "__main__":
    main()
