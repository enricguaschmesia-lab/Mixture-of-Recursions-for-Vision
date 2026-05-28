# Manim 5×5 cross-modal animation

Two-step pipeline that turns the notebook's 5×5 cross-modal grid into a Manim animation.

## 1. Install Manim

System deps (Ubuntu):

```bash
sudo apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg
```

Python dep (project uses `uv`):

```bash
uv add manim
# or, without touching pyproject:
.venv/bin/pip install manim
```

## 2. Export the 25 cells

Loads the MoR checkpoint, runs all 25 row→col generations, and dumps each cell
as a clean borderless PNG plus a `meta.json` and a depth `legend.png`.

```bash
.venv/bin/python scripts/export_5x5_cells.py \
    --config infer/mor_5000_multimodality_3r.yaml \
    --sample-id 00007 \
    --aug-idx 0 \
    --out-dir outputs/manim_5x5
```

Output:

```
outputs/manim_5x5/
  cells/
    rgb__to__rgb.png        ← GT diagonal
    rgb__to__depth.png
    ...
    scene_desc__to__scene_desc.png
  legend.png
  meta.json
```

Requires CUDA (Cosmos decoder is GPU-only).

## 3. Render the animation

```bash
# fast preview (480p):
manim -pql scripts/manim_5x5_scene.py Cross5x5Scene

# high quality (1080p, slower):
manim -pqh scripts/manim_5x5_scene.py Cross5x5Scene

# GIF instead of mp4:
manim -pql --format=gif scripts/manim_5x5_scene.py Cross5x5Scene
```

`-p` previews; output lands in `media/videos/manim_5x5_scene/.../Cross5x5Scene.{mp4,gif}`.

## Animation choreography

1. Title + subtitle fade in.
2. Empty 5×5 grid of frames sweeps in (red borders on the diagonal).
3. Row/column modality labels fade in.
4. GT diagonal cells appear in sequence.
5. Off-diagonal cells reveal **row by row** — for each input modality, the four
   cross-modal targets fade in left-to-right.
6. Depth-recursion legend appears at the bottom.

## Re-rendering for a different sample

Just rerun the export with a different `--sample-id` (any 5-digit id under
`data/clevr_dataset/test/`); the Manim scene is generic and re-reads `meta.json`.
