#!/usr/bin/env python3
"""从排样 Excel 恢复填平孔洞的底板、矩形包络及三维焊缝，并输出标注图。

load_layout_parts() 读取排样局部 U/V/N 坐标。LayoutPart.plate_faces() 可在 Python 中
构造带厚度的简化面片。排样矩形为 (0, 0) 到 (width, height)，单位为 mm。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from reconstruct_assembly_structure import (
    Assembly, INK, MUTED, PLATE, WELD, Weld, draw_model, draw_plan, sheet_records,
)

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


PROJECT_DIR = Path(__file__).resolve().parent
PACKING_EDGE = "#48658D"


@dataclass
class LayoutPart:
    ident: str
    source: str
    width: float
    height: float
    slabs: list[tuple[np.ndarray, float, float]]
    welds: list[Weld]

    def plate_faces(self) -> np.ndarray:
        return np.concatenate([extrude_outline(outline, low, high) for outline, low, high in self.slabs])


def turn(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> np.ndarray:
    edge = b - a
    offset = points - a
    return edge[0] * offset[..., 1] - edge[1] * offset[..., 0]


def triangulate_outline(outline: np.ndarray) -> np.ndarray:
    """沿逆时针外缘切出三角形，保留凹边和切口。"""
    pending = list(range(len(outline)))
    triangles = []
    epsilon = 1e-7
    while len(pending) > 3:
        for pos, index in enumerate(pending):
            first, last = pending[pos - 1], pending[(pos + 1) % len(pending)]
            a, b, c = outline[[first, index, last]]
            area2 = turn(a, b, c)
            if abs(area2) <= epsilon and (b - a) @ (b - c) <= epsilon:
                pending.pop(pos)
                break
            if area2 <= epsilon:
                continue
            others = outline[[i for i in pending if i not in (first, index, last)]]
            inside = (turn(a, b, others) >= -epsilon) & (turn(b, c, others) >= -epsilon) & (turn(c, a, others) >= -epsilon)
            if not inside.any():
                triangles.append((first, index, last))
                pending.pop(pos)
                break
        else:
            raise ValueError("底板外轮廓无法三角化")
    triangles.append(tuple(pending))
    return np.array(triangles, dtype=int)


def extrude_outline(outline: np.ndarray, low: float, high: float) -> np.ndarray:
    triangles = triangulate_outline(outline)
    bottom = np.column_stack((outline, np.full(len(outline), low)))
    top = np.column_stack((outline, np.full(len(outline), high)))
    side = np.stack((bottom, np.roll(bottom, -1, axis=0), np.roll(top, -1, axis=0), top), axis=1)
    # 与原有绘图函数使用的 3DFACE 数组兼容，三角面的末点重复第三点。
    return np.concatenate((top[triangles[:, [0, 1, 2, 2]]], bottom[triangles[:, [2, 1, 0, 0]]], side))


def load_layout_parts(path: Path) -> dict[str, LayoutPart]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        contours = defaultdict(lambda: defaultdict(list))
        for row in sheet_records(book, "底板外轮廓"):
            contours[row["编号"]][row["轮廓号"]].append((
                row["点序"], row["U (mm)"], row["V (mm)"], row["下表面N (mm)"], row["上表面N (mm)"],
            ))

        welds = defaultdict(list)
        for row in sheet_records(book, "焊缝"):
            number = int(row["类型"].removeprefix("焊缝"))
            points = np.array([[row[f"{axis}{index}"] for axis in "uvn"] for index in (1, 2)], dtype=float)
            welds[row["编号"]].append(Weld(number, row["DXF 句柄"], points))

        parts = {}
        for row in sheet_records(book, "排样尺寸"):
            ident = row["编号"]
            slabs = []
            for _, points in sorted(contours[ident].items()):
                values = np.array(sorted(points), dtype=float)
                low, high = values[0, 3:5]
                slabs.append((values[:, 1:3], float(low), float(high)))
            if not slabs:
                raise ValueError(f"{ident} 缺少底板外轮廓")
            parts[ident] = LayoutPart(
                ident, row["源 DXF"], float(row["排样U (mm)"]), float(row["排样V (mm)"]), slabs,
                sorted(welds[ident], key=lambda weld: weld.number),
            )
        return parts
    finally:
        book.close()


def render_layout(part: LayoutPart, output: Path) -> None:
    available = {font.name for font in fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available), "DejaVu Sans")
    faces = part.plate_faces()
    seams = np.array([weld.points for weld in part.welds]).reshape(-1, 2, 3)
    rings = [np.column_stack((np.vstack((outline, outline[0])), np.full(len(outline) + 1, high)))
             for outline, _, high in part.slabs]
    assembly = Assembly(part.ident, part.source, np.zeros(3), np.eye(3), float(np.ptp(faces[:, :, 2])),
                        faces, np.ones(len(faces), dtype=bool), ["ACI 3"] * len(faces), rings, part.welds)
    length_y = 0.062 + 0.034 * max(1, ceil(len(seams) / 5))
    bottom = length_y + 0.105

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(14.5, 8), facecolor="#F7FAFB")
        fig.text(0.05, 0.938, "焊接平台排样", fontsize=13, weight="bold", color=MUTED)
        fig.text(0.05, 0.868, part.ident, fontsize=33, weight="bold", color=INK)
        fig.text(0.95, 0.908, f"排样矩形  {part.width:,.2f} × {part.height:,.2f} mm",
                 ha="right", fontsize=15, color=INK)
        fig.text(0.95, 0.861, f"{len(seams)} 条焊缝记录    ·    局部 U/V/N 坐标", ha="right", fontsize=11, color=MUTED)
        fig.add_artist(Line2D([0.05, 0.95], [0.821, 0.821], transform=fig.transFigure, color="#CBD9DE", linewidth=1))
        fig.text(0.05, 0.774, "简化三维视图", fontsize=14, weight="bold", color=INK)
        fig.text(0.56, 0.774, "外轮廓与排样矩形", fontsize=14, weight="bold", color=INK)
        fig.legend(handles=[Patch(color=PLATE, label="底板"),
                            Line2D([0], [0], color=WELD, linewidth=2.5, label="焊缝中心线"),
                            Line2D([0], [0], color=PACKING_EDGE, linestyle="--", label="排样矩形")],
                   loc="center left", bbox_to_anchor=(0.044, 0.736), ncol=3, frameon=False, fontsize=10)
        model = fig.add_axes([0.027, bottom, 0.504, 0.72 - bottom])
        draw_model(model, assembly, faces, seams)
        plan = fig.add_axes([0.56, bottom + 0.027, 0.39, 0.693 - bottom])
        draw_plan(plan, faces, rings, seams, part.welds)
        plan.add_patch(Rectangle((0, 0), part.width, part.height, fill=False, edgecolor=PACKING_EDGE,
                                 linestyle="--", linewidth=1.1, zorder=3))
        x0, x1 = plan.get_xlim()
        y0, y1 = plan.get_ylim()
        plan.set(xlim=(min(x0, -0.035 * part.width), max(x1, 1.035 * part.width)),
                 ylim=(min(y0, -0.035 * part.height), max(y1, 1.035 * part.height)))

        fig.add_artist(Line2D([0.05, 0.95], [length_y + 0.027] * 2, transform=fig.transFigure,
                              color="#CBD9DE", linewidth=1))
        fig.text(0.05, length_y, "焊缝三维长度（mm）", fontsize=12, weight="bold", color=INK)
        for i, weld in enumerate(part.welds):
            row, col = divmod(i, 5)
            x, y = 0.05 + 0.181 * col, length_y - 0.03 - 0.034 * row
            fig.text(x, y, weld.label, fontsize=10, weight="bold", color=WELD)
            fig.text(x + 0.047, y, f"{weld.length_mm:,.2f}", fontsize=10, color=INK)
        fig.text(0.95, 0.021, f"源 DXF  {part.source}", ha="right", fontsize=8.5, color=MUTED)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, nargs="?", default=PROJECT_DIR / "C_layout_geometry.xlsx")
    parser.add_argument("ids", nargs="*", help="要重建的编号，省略时处理 Excel 中的全部编号")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "layout_previews")
    args = parser.parse_args()
    try:
        parts = load_layout_parts(args.workbook)
        selected = [parts[ident] for ident in dict.fromkeys(args.ids)] if args.ids else list(parts.values())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for part in selected:
            output = args.output_dir / f"{part.ident}.png"
            render_layout(part, output)
            print(f"{part.ident}：{len(part.slabs)} 个底板外轮廓、{len(part.welds)} 条焊缝，保存至 {output.resolve()}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"重建失败：{error}\n")


if __name__ == "__main__":
    main()
