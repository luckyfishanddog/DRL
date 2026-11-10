#!/usr/bin/env python3
"""读取提取的结构 Excel，恢复原始三维面片、底板闭合轮廓和焊缝并输出 PNG。

load_assemblies() 返回的 faces、contours 和 welds.points 均为 DXF 世界坐标（mm）。
图片用正交的底板局部 U/V/N 坐标显示，焊缝长度按三维端点距离计算。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from math import ceil, dist
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from openpyxl import load_workbook


PROJECT_DIR = Path(__file__).resolve().parent
PLATE = "#65A782"
PLATE_EDGE = "#34684F"
WELD = "#C52233"
INK = "#1C3441"
MUTED = "#4D6370"
STEEL = {"ACI 1": "#637B92", "ACI 3": "#8CAB99", "ACI 4": "#8AAFC9"}


@dataclass
class Weld:
    number: int
    handle: str
    points: np.ndarray

    @property
    def label(self) -> str:
        return f"W{self.number:02d}"

    @property
    def length_mm(self) -> float:
        return dist(*self.points)


@dataclass
class Assembly:
    ident: str
    source: str
    origin: np.ndarray
    basis: np.ndarray
    thickness_mm: float
    faces: np.ndarray
    base_mask: np.ndarray
    colors: list[str]
    contours: list[np.ndarray]
    welds: list[Weld]

    def local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.basis


def sheet_records(book, name):
    rows = iter(book[name].values)
    headers = next(rows)
    return (dict(zip(headers, row)) for row in rows if row[0] is not None)


def entity_points(row, count: int) -> np.ndarray:
    return np.array([[float(row[f"{axis}{i}"]) for axis in "xyz"] for i in range(1, count + 1)])


def load_assemblies(path: Path) -> dict[str, Assembly]:
    """从 Excel 恢复几何。字典按小组立编号索引，面片数组形状为 (面片数, 4, 3)。"""
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        faces, base_masks, colors = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in sheet_records(book, "结构面片"):
            ident = row["编号"]
            faces[ident].append(entity_points(row, 4))
            base_masks[ident].append(row["部件"] == "底板")
            colors[ident].append(row["颜色"])

        loops = defaultdict(lambda: defaultdict(list))
        for row in sheet_records(book, "底板轮廓"):
            loops[row["编号"]][row["轮廓号"]].append((row["点序"], (row["x"], row["y"], row["z"])))

        welds = defaultdict(list)
        for row in sheet_records(book, "焊缝"):
            number = int(row["类型"].removeprefix("焊缝"))
            welds[row["编号"]].append(Weld(number, row["DXF 句柄"], entity_points(row, 2)))

        result = {}
        for row in sheet_records(book, "小组立"):
            ident = row["编号"]
            origin = np.array([row[f"参考{axis} (mm)"] for axis in "XYZ"], dtype=float)
            u = np.array([row[f"U{axis}"] for axis in "xyz"], dtype=float)
            v = np.array([row[f"V{axis}"] for axis in "xyz"], dtype=float)
            basis = np.column_stack((u, v, np.cross(u, v)))
            contours = [np.array([point for _, point in sorted(points)], dtype=float)
                        for _, points in sorted(loops[ident].items())]
            if not any(base_masks[ident]) or not contours:
                raise ValueError(f"{ident} 缺少底板面片或轮廓")
            result[ident] = Assembly(
                ident, row["源 DXF"], origin, basis, float(row["底板法向跨度 (mm)"]),
                np.array(faces[ident]), np.array(base_masks[ident]), colors[ident], contours,
                sorted(welds[ident], key=lambda weld: weld.number),
            )
        return result
    finally:
        book.close()


def rasterize_mesh(triangles: np.ndarray, colors: np.ndarray, low: np.ndarray, high: np.ndarray):
    """用正交视图的逐像素深度处理薄板与加强件的遮挡。"""
    scale = min(np.array([1440, 800]) / (high - low))
    width, height = np.ceil((high - low) * scale).astype(int)
    depth_buffer = np.full((height, width), -np.inf)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    pixel_triangles = (triangles[:, :, :2] - low) * scale
    rgb = np.rint(255 * colors).astype(np.uint8)
    for points, depth, color in zip(pixel_triangles, triangles[:, :, 2], rgb):
        x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int), (width, height))
        a, b, c = points
        u, v = b - a, c - a
        determinant = u[0] * v[1] - u[1] * v[0]
        if abs(determinant) < 1e-12 or x0 == x1 or y0 == y1:
            continue
        x = np.arange(x0, x1)[None, :] + 0.5 - a[0]
        y = np.arange(y0, y1)[:, None] + 0.5 - a[1]
        s = (x * v[1] - y * v[0]) / determinant
        t = (u[0] * y - u[1] * x) / determinant
        z = depth[0] + s * (depth[1] - depth[0]) + t * (depth[2] - depth[0])
        stored = depth_buffer[y0:y1, x0:x1]
        visible = (s >= -1e-10) & (t >= -1e-10) & (s + t <= 1 + 1e-10) & (z > stored)
        stored[visible] = z[visible]
        pixels = image[y0:y1, x0:x1]
        pixels[visible] = (*color, 255)
    right, top = low + np.array([width, height]) / scale
    return image, (low[0], right, low[1], top)


def draw_model(ax, assembly: Assembly, faces: np.ndarray, seams: np.ndarray) -> None:
    triangles = np.concatenate((faces[:, :3], faces[:, [0, 2, 3]]))
    a, b, c = triangles.transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    area2 = np.linalg.norm(normals, axis=1)
    valid = area2 > 0
    light = np.array([-0.38, -0.45, 0.81])
    shade = 0.64 + 0.36 * np.abs((normals[valid] / area2[valid, None]) @ light)
    colors = np.array([to_rgb(PLATE if base else STEEL.get(color, "#7E919F"))
                       for base, color in zip(assembly.base_mask, assembly.colors)])
    triangle_colors = np.tile(colors, (2, 1))[valid] * shade[:, None]
    elev, azim = np.radians([32, -64])
    forward = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    right = np.array([-np.sin(azim), np.cos(azim), 0])
    view = np.column_stack((right, np.cross(forward, right), forward))
    view_triangles, view_seams = triangles[valid] @ view, seams @ view
    points = np.concatenate((view_triangles.reshape(-1, 3), view_seams.reshape(-1, 3)))[:, :2]
    margin = np.ptp(points, axis=0).max() * 0.04
    image, bounds = rasterize_mesh(view_triangles, triangle_colors, points.min(0) - margin, points.max(0) + margin)
    ax.imshow(image, origin="lower", extent=bounds, aspect="equal", interpolation="antialiased", zorder=1)
    if len(view_seams):
        segments = view_seams[:, :, :2]
        ax.add_collection(LineCollection(segments, colors="white", linewidths=4.0, capstyle="round", zorder=3))
        ax.add_collection(LineCollection(segments, colors=WELD, linewidths=2.45, capstyle="round", zorder=4))
        zero = np.array([weld.length_mm == 0 for weld in assembly.welds])
        if zero.any():
            ax.scatter(*segments[zero, 0].T, marker="x", color=WELD, s=32, zorder=5)
    ax.set(xlim=bounds[:2], ylim=bounds[2:])
    ax.set_axis_off()


def draw_plan(ax, green: np.ndarray, rings: list[np.ndarray], seams: np.ndarray, welds: list[Weld]) -> None:
    ax.add_collection(PolyCollection(green[:, :, :2], facecolors="#DCEEE3", edgecolors="none", zorder=1))
    ax.add_collection(LineCollection([ring[:, :2] for ring in rings], colors=PLATE_EDGE, linewidths=1.0, zorder=2))
    ax.add_collection(LineCollection(seams[:, :, :2], colors=WELD, linewidths=2.4, capstyle="round", zorder=4))
    positions = seams[:, :, :2].mean(axis=1)
    ax.scatter(*positions.T, s=11, color=WELD, zorder=5)
    points = np.concatenate((green.reshape(-1, 3), seams.reshape(-1, 3)))[:, :2]
    low, high = points.min(axis=0), points.max(axis=0)
    width, height = high - low
    ax.set(xlim=(low[0] - 0.21 * width, high[0] + 0.21 * width),
           ylim=(low[1] - 0.16 * height, high[1] + 0.16 * height), xlabel="U (mm)", ylabel="V (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_axisbelow(True)
    ax.grid(color="#E4ECEF", linewidth=0.65)
    for spine in ax.spines.values():
        spine.set_color("#CBD9DE")

    left = positions[:, 0] <= (low[0] + high[0]) / 2
    for side, mask in ((-1, left), (1, ~left)):
        order = sorted(np.flatnonzero(mask), key=lambda i: (positions[i, 1], positions[i, 0], i))
        x = low[0] - 0.085 * width if side < 0 else high[0] + 0.085 * width
        slots = positions[order, 1] if len(order) <= 1 else np.linspace(low[1], high[1], len(order))
        for index, y in zip(order, slots):
            ax.annotate(welds[index].label, positions[index], xytext=(x, y), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=WELD, zorder=6,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#F0C0C4", "lw": 0.7},
                        arrowprops={"arrowstyle": "-", "color": "#9A5960", "lw": 0.65, "shrinkA": 1, "shrinkB": 1})


def render_assembly(assembly: Assembly, output: Path) -> None:
    available = {font.name for font in fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in available), "DejaVu Sans")
    faces = assembly.local(assembly.faces)
    rings = [assembly.local(np.vstack((ring, ring[0]))) for ring in assembly.contours]
    seams = assembly.local(np.array([weld.points for weld in assembly.welds]).reshape(-1, 2, 3))
    span = np.ptp(faces.reshape(-1, 3), axis=0)
    rows = max(1, ceil(len(seams) / 5))
    length_title_y = 0.066 + 0.034 * rows
    panel_bottom = length_title_y + 0.10

    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(15, 8.55), facecolor="#F7FAFB")
        fig.text(0.05, 0.94, "小组立三维重建", fontsize=13, weight="bold", color=MUTED)
        fig.text(0.05, 0.875, assembly.ident, fontsize=34, weight="bold", color=INK)
        dimension = " × ".join(f"{value:,.2f}" for value in span)
        fig.text(0.95, 0.91, f"外形 U × V × N  {dimension} mm", ha="right", fontsize=15, color=INK)
        fig.text(0.95, 0.866, f"底板法向跨度 {assembly.thickness_mm:.2f} mm    ·    {len(seams)} 条焊缝记录",
                 ha="right", fontsize=11, color=MUTED)
        fig.add_artist(Line2D([0.05, 0.95], [0.826, 0.826], transform=fig.transFigure, color="#CBD9DE", linewidth=1))

        fig.text(0.05, 0.779, "等比例三维视图", fontsize=14, weight="bold", color=INK)
        fig.text(0.565, 0.779, "底板轮廓与焊缝投影", fontsize=14, weight="bold", color=INK)
        fig.legend(handles=[Patch(color=PLATE, label="底板"), Patch(color=STEEL["ACI 1"], label="其他构件"),
                            Line2D([0], [0], color=WELD, linewidth=2.5, label="焊缝")],
                   loc="center left", bbox_to_anchor=(0.044, 0.744), ncol=3, frameon=False, fontsize=10)
        model = fig.add_axes([0.028, panel_bottom, 0.50, 0.74 - panel_bottom])
        model.set_facecolor(fig.get_facecolor())
        draw_model(model, assembly, faces, seams)
        plan = fig.add_axes([0.565, panel_bottom + 0.03, 0.385, 0.71 - panel_bottom])
        draw_plan(plan, faces[assembly.base_mask], rings, seams, assembly.welds)

        fig.add_artist(Line2D([0.05, 0.95], [length_title_y + 0.027] * 2,
                              transform=fig.transFigure, color="#CBD9DE", linewidth=1))
        fig.text(0.05, length_title_y, "焊缝三维长度（mm）", fontsize=12, weight="bold", color=INK)
        for i, weld in enumerate(assembly.welds):
            row, col = divmod(i, 5)
            x, y = 0.05 + 0.181 * col, length_title_y - 0.031 - 0.034 * row
            fig.text(x, y, weld.label, fontsize=10, weight="bold", color=WELD)
            fig.text(x + 0.044, y, f"{weld.length_mm:,.2f}", fontsize=10, color=INK)
        fig.text(0.95, 0.023, f"源 DXF  {assembly.source}    ·    局部 U/V/N 正交视图",
                 ha="right", fontsize=8.5, color=MUTED)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="提取脚本生成的 Excel")
    parser.add_argument("ids", nargs="*", help="要重建的小组立编号，省略时处理工作簿中的全部编号")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "structure_previews")
    args = parser.parse_args()
    try:
        assemblies = load_assemblies(args.workbook)
        selected = [assemblies[ident] for ident in dict.fromkeys(args.ids)] if args.ids else list(assemblies.values())
        if not selected:
            raise ValueError("工作簿里没有小组立")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for assembly in selected:
            output = args.output_dir / f"{assembly.ident}.png"
            render_assembly(assembly, output)
            print(f"{assembly.ident}：重建 {len(assembly.faces)} 个面片、{len(assembly.welds)} 条焊缝，保存至 {output.resolve()}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"重建失败：{error}\n")


if __name__ == "__main__":
    main()
