"""The universal 32-colour palette and each task's class -> colour assignment (A2, decision 6).

Decision 6 of the A2 step plan (docs/plans/a2_segmentation.md):
  - 32 colours, greedy farthest point in RGB, from black and white;
  - white is void or no data, never predicted; black is the first class colour;
  - LULC's 9 classes take colours 1-9;
  - a downstream class that matches a LULC class reuses its colour (CloudSen12's thick cloud takes
    LULC's clouds colour);
  - the rest take the next unused colours in greedy order, from colour 10 on.

Index layout of UNIVERSAL: 0 = white (void / no data), 1 = black, 2..31 = the greedy picks in order.
The greedy search is Step 0's (`rgb_map_ceiling._greedy`, the `distinct` palette: a 6-level RGB grid),
so LULC's colours here are exactly Step 0's `distinct` LULC palette.

Assignments are per task and injective over the task's scored classes. Colour 0 is the void colour
and only the ignored class may take it. A task's own classes never share a colour (Z1 refuses it).

Training renders for a dedicated map tokenizer use `shuffled_assignment`: a random injective
class -> colour map over colours 1..31 per map, so the tokenizer learns geometry and the 32 colours,
never one dataset's classes. Reads always use the task's fixed assignment.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

N_COLOURS = 32
VOID_COLOUR = 0
GRID_LEVELS = 6            # Step 0's `distinct` palette: farthest point on a 6^3 RGB grid


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] (..., 3) -> CIELAB (D65). Step 0's, unchanged (`rgb_map_ceiling.py`)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def greedy(n: int, levels: int = GRID_LEVELS, space: str = "rgb") -> np.ndarray:
    """Farthest-point colours on a levels^3 RGB grid, from black and white. Step 0's `_greedy`,
    unchanged: returns [black, white, pick 3, pick 4, ...]."""
    lv = np.linspace(0, 1, levels)
    grid = np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
    emb = _srgb_to_lab(grid) if space == "lab" else grid
    chosen = [0, len(grid) - 1]                       # black, white
    while len(chosen) < n:
        d = ((emb[:, None] - emb[chosen][None]) ** 2).sum(-1).min(1)
        chosen.append(int(d.argmax()))
    return grid[chosen]


def _universal() -> np.ndarray:
    g = greedy(N_COLOURS)
    return np.concatenate([g[1:2], g[0:1], g[2:]])   # 0 white, 1 black, 2.. greedy order


UNIVERSAL = _universal()                              # (32, 3) float64 in [0, 1]

LULC_NAMES = ("no data", "water", "trees", "flooded vegetation", "crops", "built area", "bare ground",
              "snow/ice", "clouds", "rangeland")
PASTIS_NAMES = ("background", "meadow", "soft winter wheat", "corn", "winter barley", "winter rapeseed",
                "spring barley", "sunflower", "grapevine", "beet", "winter triticale",
                "winter durum wheat", "fruits, vegetables, flowers", "potatoes", "leguminous fodder",
                "soybeans", "orchard", "mixed cereal", "sorghum", "void label")
CLOUDSEN12_NAMES = ("clear", "thick cloud", "thin cloud", "cloud shadow")


@dataclass(frozen=True)
class Task:
    """One map task: its classes, the ignored class (None if every class is scored), the native
    grid it is scored on, and its fixed class -> universal colour assignment."""
    name: str
    class_names: tuple
    ignore: int | None
    native: int                                       # px of the scored grid (CloudSen12: the tile)
    assign: tuple                                     # class id -> index into UNIVERSAL
    lulc_matches: dict = field(default_factory=dict)  # class id -> LULC class whose colour it reuses

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def scored(self) -> list[int]:
        return [c for c in range(self.n_classes) if c != self.ignore]


def _assign(n_classes: int, ignore: int | None, matches: dict[int, int]) -> tuple:
    """Matched classes take their LULC class's colour; the rest take colours 10, 11, ... in class
    order, skipping any colour a match already took; the ignored class takes the void colour."""
    out = [None] * n_classes
    for c, lulc_class in matches.items():
        out[c] = lulc_class
    nxt = 10
    for c in range(n_classes):
        if c == ignore:
            out[c] = VOID_COLOUR
        elif out[c] is None:
            while nxt in out:
                nxt += 1
            out[c] = nxt
            nxt += 1
    return tuple(out)


TASKS: dict[str, Task] = {
    # LULC: class c -> colour c (0 = no data = white, 1 = water = black), decision 6.
    "lulc": Task("lulc", LULC_NAMES, ignore=0, native=224, assign=tuple(range(10))),
    # PASTIS: no class matches one LULC class (background is every non-agricultural cover; the crop
    # types are all LULC "crops"), so classes 0..18 take colours 10..28 and void (19) is white.
    "pastis": Task("pastis", PASTIS_NAMES, ignore=19, native=128, assign=_assign(20, 19, {})),
    # CloudSen12: thick cloud reuses LULC's clouds (8); clear, thin cloud and shadow take 10, 11, 12.
    # GEO-Bench-2's CloudSen12 has no void class: all four are scored.
    "cloudsen12": Task("cloudsen12", CLOUDSEN12_NAMES, ignore=None, native=512,
                       assign=_assign(4, None, {1: 8}), lulc_matches={1: 8}),
}


class PaletteError(ValueError):
    pass


def validate(colours: np.ndarray, assign, ignore: int | None, n_classes: int | None = None,
             void_index: int | None = VOID_COLOUR) -> None:
    """Z1's refusals. The assignment must cover every class, map the ignored class (and only it) to
    the void colour, give every scored class its own colour, and no two colours a task uses may be
    equal (in RGB to 1e-9). `void_index` None takes the ignored class's own index as the void colour
    (Step 0's per-dataset palettes)."""
    assign = list(assign)
    if n_classes is not None and len(assign) != n_classes:
        raise PaletteError(f"assignment covers {len(assign)} classes, task has {n_classes}")
    if any(not (0 <= a < len(colours)) for a in assign):
        raise PaletteError("assignment points outside the palette")
    scored = [c for c in range(len(assign)) if c != ignore]
    if void_index is None:
        void_index = assign[ignore] if ignore is not None else -1
    if ignore is not None and assign[ignore] != void_index:
        raise PaletteError("the ignored class must take the void colour")
    if any(assign[c] == void_index for c in scored):
        raise PaletteError("a scored class takes the void colour, which is never predicted")
    if len({assign[c] for c in scored}) != len(scored):
        raise PaletteError("two scored classes share a palette index")
    used = sorted(set(assign))
    rgb = np.asarray(colours, dtype=np.float64)[used]
    d = np.sqrt(((rgb[:, None] - rgb[None]) ** 2).sum(-1))
    d[np.diag_indices(len(used))] = np.inf
    if d.min() < 1e-9:
        raise PaletteError("two colours the task uses are equal")


def task_colours(task: Task, colours: np.ndarray = UNIVERSAL) -> np.ndarray:
    """(n_classes, 3): the RGB of each class under the task's assignment."""
    return np.asarray(colours, dtype=np.float64)[list(task.assign)]


def min_distance(task: Task, colours: np.ndarray = UNIVERSAL) -> float:
    """Smallest RGB distance between two scored classes' colours."""
    rgb = task_colours(task, colours)[task.scored]
    d = np.sqrt(((rgb[:, None] - rgb[None]) ** 2).sum(-1))
    d[np.diag_indices(len(rgb))] = np.inf
    return float(d.min())


def render(labels: torch.Tensor, assign, size: int, colours: np.ndarray = UNIVERSAL) -> torch.Tensor:
    """Class maps (B, H, W) int -> RGB (B, 3, size, size) float32 in [0, 1], nearest upsampling, on
    the labels' device. `assign` is a class -> palette index table, (n_classes,) or per map (B, n)."""
    col = torch.as_tensor(np.asarray(colours), dtype=torch.float32, device=labels.device)
    a = torch.as_tensor(np.asarray(assign), dtype=torch.long, device=labels.device)
    lab = labels.long()
    if a.ndim == 1:
        idx = a[lab]
    else:                                               # one assignment per map
        idx = torch.gather(a, 1, lab.reshape(len(lab), -1)).reshape(lab.shape)
    x = col[idx].permute(0, 3, 1, 2)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="nearest")
    return x.contiguous()


def shuffled_assignment(n_maps: int, n_classes: int, ignore: int | None,
                        generator: torch.Generator | None = None, device="cpu") -> torch.Tensor:
    """(n_maps, n_classes): per map, a random injective class -> colour assignment over colours
    1..31, the ignored class on the void colour. For training renders only."""
    if n_classes - (ignore is not None) > N_COLOURS - 1:
        raise PaletteError("more scored classes than predictable colours")
    r = torch.rand(n_maps, N_COLOURS - 1, generator=generator, device=device)
    perm = torch.argsort(r, 1)[:, :n_classes] + 1      # distinct colours in 1..31
    if ignore is not None:
        perm[:, ignore] = VOID_COLOUR
    return perm


def palette_record(colours: np.ndarray = UNIVERSAL) -> dict:
    """What a frozen tokenizer records: the colours, each task's assignment, and one hash."""
    rec = {"colours": np.asarray(colours).round(6).tolist(),
           "tasks": {k: {"assign": list(t.assign), "ignore": t.ignore, "classes": list(t.class_names),
                         "lulc_matches": t.lulc_matches} for k, t in TASKS.items()}}
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


def step0_distinct(task: Task) -> tuple[np.ndarray, tuple]:
    """Step 0's per-dataset `distinct` palette, for the regression anchor only: (colours, identity
    assignment). The ignored class is white, class 0 (or 1 when 0 is ignored) black, the rest the
    greedy picks in class order (`rgb_map_ceiling.palette('distinct', ...)`, unchanged)."""
    n, ignore = task.n_classes, task.ignore
    g = greedy(n)
    black, white, rest = g[0], g[1], list(g[2:])
    out, k = np.zeros((n, 3)), 0
    first_scored = 0 if ignore != 0 else 1
    for c in range(n):
        if c == ignore:
            out[c] = white
        elif c == first_scored:
            out[c] = black
        else:
            out[c] = rest[k]
            k += 1
    return out, tuple(range(n))


for _t in TASKS.values():                             # refuse a bad table at import
    validate(UNIVERSAL, _t.assign, _t.ignore, _t.n_classes)
