"""Manim scene: 5x5 cross-modal grid, slide-friendly (~15s, no title).

Designed to be embedded in a slide — no title, no sample tag, just the
grid + phase caption + legend. Walks through:

  1. Empty 5x5 frame + row/col modality labels.
  2. "Diagonal = ground truth" → diag GT cells fade in.
  3. Phase caption flips per row ("Given X, MoR predicts ..."); a
     yellow rectangle highlights the active row while its 4
     off-diagonal predictions appear left-to-right.
  4. Inline legend explaining the MoR recursion-depth color mapping.

Render:
    manim -ql scripts/manim_5x5_scene.py Cross5x5Scene   # 480p preview
    manim -qh scripts/manim_5x5_scene.py Cross5x5Scene   # 1080p
"""

from __future__ import annotations

import json
from pathlib import Path

from manim import (
    Scene, Text, ImageMobject, Square, VGroup, Group,
    SurroundingRectangle, Transform, FadeIn, FadeOut, LaggedStartMap,
    RIGHT,
    RED, GREY_B, GREY_D, WHITE, YELLOW,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "outputs" / "manim_5x5"

# Friendly modality names. Two variants because col labels have less
# horizontal room than the phase caption.
PRETTY = {
    "tok_rgb@256": "RGB",
    "tok_depth@256": "Depth",
    "tok_normal@256": "Normal",
    "caption": "Caption",
    "scene_desc": "Scene desc.",
}
SHORT_LABEL = {
    "tok_rgb@256": "RGB",
    "tok_depth@256": "Depth",
    "tok_normal@256": "Normal",
    "caption": "Caption",
    "scene_desc": "Scene",
}


class Cross5x5Scene(Scene):
    def construct(self):
        meta = json.loads((DATA_DIR / "meta.json").read_text())
        mods = meta["modalities"]
        n = len(mods)
        short = meta["short_name"]
        depth_colors = meta["depth_colors"]
        num_recursions = meta["num_recursions"]

        # -------- layout (manim default frame: 14.22w x 8h) --------
        # No title — grid grows to fill the space. y-budget top→bottom:
        #   3.50  phase caption
        #   3.00  col header ("predicted target →")
        #   2.70  col labels (RGB / Depth / ...)
        #   2.10  top row of cells (cell center)
        #  -2.78  bottom row of cells (4*(cell+gap) below)
        #  -3.65  legend
        # Row labels eat ~1.0 unit on the left, so we shift the grid right.
        cell_side = 1.15
        gap = 0.07
        top_row_y = 2.10
        grid_center_x = 0.55
        grid_w = n * cell_side + (n - 1) * gap
        grid_left = grid_center_x - grid_w / 2 + cell_side / 2

        def cc(i: int, j: int) -> list[float]:
            x = grid_left + j * (cell_side + gap)
            y = top_row_y - i * (cell_side + gap)
            return [x, y, 0.0]

        # Phase caption — Transform-target during the animation so the
        # placement is reused each phase. Initialized with a single
        # space so the empty Text has a stable position.
        phase_y = 3.70
        phase = Text(" ", font_size=24, color=WHITE)
        phase.move_to([grid_center_x, phase_y, 0.0])

        def set_phase(new_text: str, color=WHITE):
            target = Text(new_text, font_size=24, color=color)
            target.move_to([grid_center_x, phase_y, 0.0])
            self.play(Transform(phase, target), run_time=0.4)

        # -------- grid frames + cells --------
        frames: dict[tuple[int, int], Square] = {}
        cells: dict[tuple[int, int], ImageMobject] = {}
        for i, src in enumerate(mods):
            for j, tgt in enumerate(mods):
                name = f"{short[src]}__to__{short[tgt]}"
                info = meta["cells"][name]
                is_gt = info["is_gt"]
                frame = Square(
                    side_length=cell_side,
                    stroke_width=3.0 if is_gt else 1.0,
                    color=RED if is_gt else GREY_D,
                ).move_to(cc(i, j))
                img = ImageMobject(str(DATA_DIR / info["file"]))
                img.height = cell_side - 0.04
                img.move_to(cc(i, j))
                frames[(i, j)] = frame
                cells[(i, j)] = img

        # -------- row / col labels --------
        col_labels_y = 3.00
        col_header_y = 3.30
        row_label_x = grid_left - cell_side / 2 - 0.60

        # Plain (non-bold, non-italic) Text avoids Pango letter-spacing
        # artifacts. Hierarchy comes from color (WHITE vs GREY_B) and
        # font size, not weight/slant.
        col_labels = VGroup()
        for j, tgt in enumerate(mods):
            t = Text(SHORT_LABEL[tgt], font_size=19, color=WHITE)
            t.move_to([cc(0, j)[0], col_labels_y, 0.0])
            col_labels.add(t)
        col_header = Text(
            "predicted target",
            font_size=16,
            color=GREY_B,
        )
        col_header.move_to([grid_center_x, col_header_y, 0.0])

        row_labels = VGroup()
        for i, src in enumerate(mods):
            t = Text(SHORT_LABEL[src], font_size=19, color=WHITE)
            t.move_to([row_label_x, cc(i, 0)[1], 0.0])
            row_labels.add(t)
        row_header = Text(
            "input",
            font_size=16,
            color=GREY_B,
        )
        row_header.move_to([row_label_x, col_labels_y, 0.0])

        # -------- inline legend (Manim-built) --------
        # Single horizontal strip: prefix + N chips with inline labels +
        # red-border-=-GT marker. Keeps the legend short enough to clear
        # the grid's bottom row.
        chip_side = 0.28
        legend_prefix = Text(
            "color = MoR recursion depth:",
            font_size=17,
            color=GREY_B,
        )
        chip_items: list[VGroup] = []
        for d, hex_color in enumerate(depth_colors[:num_recursions]):
            chip = Square(
                side_length=chip_side,
                stroke_width=0.8,
                color=GREY_B,
                fill_color=hex_color,
                fill_opacity=1.0,
            )
            lab = Text(f"r={d + 1}", font_size=16, color=WHITE)
            chip_items.append(VGroup(chip, lab).arrange(RIGHT, buff=0.08))
        chip_strip = VGroup(*chip_items).arrange(RIGHT, buff=0.28)

        gt_chip = Square(
            side_length=chip_side,
            stroke_width=2.2,
            color=RED,
            fill_opacity=0.0,
        )
        gt_lab = Text("= ground truth", font_size=16, color=WHITE)
        gt_marker = VGroup(gt_chip, gt_lab).arrange(RIGHT, buff=0.10)

        legend = VGroup(legend_prefix, chip_strip, gt_marker).arrange(
            RIGHT, buff=0.40,
        )
        legend.move_to([0.0, -3.65, 0.0])

        # ================= animation (~15s) =================
        # 1. Empty grid + headers (one combined fade-in).
        all_frames = VGroup(*[frames[(i, j)] for i in range(n) for j in range(n)])
        labels = VGroup(col_header, row_header, col_labels, row_labels)
        self.add(phase)
        self.play(
            LaggedStartMap(FadeIn, all_frames, lag_ratio=0.012),
            FadeIn(labels),
            run_time=0.8,
        )

        # 2. Diagonal = ground truth.
        set_phase("Diagonal = ground truth inputs", color=RED)
        diag_highlight = SurroundingRectangle(
            VGroup(*[frames[(k, k)] for k in range(n)]),
            color=RED,
            stroke_width=2.0,
            buff=0.05,
        )
        diag = Group(*[cells[(k, k)] for k in range(n)])
        self.play(
            FadeIn(diag_highlight),
            LaggedStartMap(FadeIn, diag, lag_ratio=0.12),
            run_time=1.1,
        )
        self.wait(0.35)
        self.play(FadeOut(diag_highlight), run_time=0.25)

        # 3. Row-by-row predictions (5 rows × ~1.4s each ≈ 7.0s).
        for i, src in enumerate(mods):
            set_phase(
                f"Given {PRETTY[src]}, MoR predicts the four other modalities",
                color=YELLOW,
            )
            row_box = SurroundingRectangle(
                VGroup(*[frames[(i, j)] for j in range(n)]),
                color=YELLOW,
                stroke_width=2.5,
                buff=0.04,
            )
            row_cells = Group(*[cells[(i, j)] for j in range(n) if j != i])
            self.play(
                FadeIn(row_box),
                LaggedStartMap(FadeIn, row_cells, lag_ratio=0.12),
                run_time=0.9,
            )
            self.wait(0.20)
            self.play(FadeOut(row_box), run_time=0.18)

        # 4. Legend.
        set_phase(
            "Color hue = which recursion depth generated each token",
            color=WHITE,
        )
        self.play(FadeIn(legend), run_time=0.6)
        self.wait(2.2)
