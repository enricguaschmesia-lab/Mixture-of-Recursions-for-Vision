"""Several of agent B's runs back to back in ONE RCP job (A2 Step 1), so a slot never idles overnight.

    python -X faulthandler -u -m eo.experiments.segmentation.step1_B_rcp_chain --chain <name> --out-root <$L/a2seg/out/b>

Each link is one `step1_B_rcp_run` (train to the end of its schedule, then read best.pt with the controls); the
chains are defined here, so they are frozen with the code snapshot (sync_code.sh). Restartable from the top: a
finished link exits at once (done.json, read_best.json with its controls). A link that fails is logged and the
chain moves on to the next one.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

# wave 1's recipe (b224_4375, b256_4375), less the mixture, which each link names
REC = ("--lr 5e-4 --warmup 2000 --ema 0.9995 --pastis-mosaic 0.5 --eval-every 2500 "
       "--min-stable 30000 --max-stable 150000 --plateau-gain 0.003 --decay-frac 0.2 --min-decay 8000 "
       "--precision bf16 --data-gpu --expect-gpu A100")
MIX = "--comp 12,10,10"          # wave 1's: PASTIS 12, CloudSen12 10, LULC 10 maps per step
MIX_L = "--comp 8,10,14"         # LULC-heavier: LULC is the binding bar, PASTIS has the margin
# a stricter flattening rule (argparse keeps the last value, so these override REC's): the stable phase lasts at
# least 45k steps and until the mean monitor mIoU gains under 0.15 points over the last quarter (the brief's 0.3
# fired at 30-37.5k while monitor LULC still rose ~0.1 point per 2,500 steps), at most 100k; decay 20% of it.
LONG = "--min-stable 45000 --plateau-gain 0.0015 --max-stable 100000"

CHAINS: dict[str, list[dict]] = {
    # wave 2, slot freed by b224_4375: the 15,360-code FSQ (8-8-8-6-5) at 196 tokens, then the plan's 1,000 codes
    # at the better size (256)
    "w2a": [{"run": "b224_15360", "train": f"--size 224 --levels 8,8,8,6,5 {MIX} {REC}"},
            {"run": "b256_1000", "train": f"--size 256 --levels 8,5,5,5 {MIX} {REC}", "read": "--native256"}],
    # wave 2, slot freed by b256_4375: 15,360 codes at 256 tokens, then the same with the LULC-heavier mixture
    # (never submitted: replaced by w2c once wave 1's reads showed LULC's errors sit on boundaries, 2026-11-18)
    "w2b": [{"run": "b256_15360", "train": f"--size 256 --levels 8,8,8,6,5 {MIX} {REC}", "read": "--native256"},
            {"run": "b256_15360_mixl", "train": f"--size 256 --levels 8,8,8,6,5 {MIX_L} {REC}",
             "read": "--native256"}],
    # wave 2, slot freed by b256_4375: one lever at a time at 256 px against b256_4375 (LULC read 0.9439, bar 0.95;
    # boundary accuracy 0.65 against 0.995 interior): 15,360 codes; boundary pixels weighted 3x in the loss;
    # the LULC-heavier mixture
    "w2c": [{"run": "b256_15360", "train": f"--size 256 --levels 8,8,8,6,5 {MIX} {REC}", "read": "--native256"},
            {"run": "b256_4375_bw2", "train": f"--size 256 --levels 7,5,5,5,5 {MIX} --boundary-weight 2 {REC}",
             "read": "--native256"},
            {"run": "b256_4375_mixl", "train": f"--size 256 --levels 7,5,5,5,5 {MIX_L} {REC}",
             "read": "--native256"}],
    # wave 3, slot freed by w2a: the 15,360-code design (ahead of 4,375 on monitor at both sizes, by 0.5-0.9 points
    # of mean mIoU at matched steps) trained longer, at 256 then 224 px; wave 1's mixture and plain loss (the
    # boundary-weighted run trailed b256_4375 on monitor at matched steps)
    "w3a": [{"run": "b256_15360_long", "train": f"--size 256 --levels 8,8,8,6,5 {MIX} {REC} {LONG}",
             "read": "--native256"},
            {"run": "b224_15360_long", "train": f"--size 224 --levels 8,8,8,6,5 {MIX} {REC} {LONG}"}],
    # wave 3, slot freed by w2c: does the codebook trend continue? Monitor rose at every step 1,000 -> 4,375 ->
    # 15,360 (at 256 px, mean monitor mIoU 0.9218 -> 0.9345 -> 0.9429). 64,000 codes (FSQ 8-8-8-5-5-5, Cosmos's
    # size), wave 1's recipe, at 224 then 256 px: an exploration beyond the brief's two codebooks.
    "w3c": [{"run": "b224_64000", "train": f"--size 224 --levels 8,8,8,5,5,5 {MIX} {REC}"},
            {"run": "b256_64000", "train": f"--size 256 --levels 8,8,8,5,5,5 {MIX} {REC}", "read": "--native256"}],
    # wave 4, the 196-token slot: 64,000 codes (ahead of 15,360 on monitor at matched steps, +0.5 points of mean mIoU
    # at 5k-7.5k) trained longer (b256_15360_long: monitor LULC 0.9475 at 52.5k against b256_15360's 0.9430 at 40.5k)
    "w4": [{"run": "b224_64000_long", "train": f"--size 224 --levels 8,8,8,5,5,5 {MIX} {REC} {LONG}"}],
    # wave 5, slot freed by w3a: does the plan's 4,375-code FSQ reach the 256-token bar when trained as long; and a
    # seed-1 replicate of the 256-token 15,360-code long run (replicate noise: ~0.3-0.4 points of mean monitor mIoU
    # at matched steps between b256_15360 and b256_15360_long before their schedules diverged)
    "w5": [{"run": "b256_4375_long", "train": f"--size 256 --levels 7,5,5,5,5 {MIX} {REC} {LONG}",
            "read": "--native256"},
           {"run": "b256_15360_long_s1", "train": f"--size 256 --levels 8,8,8,6,5 {MIX} {REC} {LONG} --seed 1",
            "read": "--native256"}],
    # wave 6, slot freed by w4: model capacity, the one lever not yet tried for the 196-token slot. Widths x1.5
    # (96-192-384-768) and attention width 768: 117.2M parameters against 52.2M; 15,360 codes; long schedule.
    "w6": [{"run": "b224_15360_wide_long",
            "train": f"--size 224 --levels 8,8,8,6,5 --widths 96,192,384,768 --attn-dim 768 {MIX} {REC} {LONG}"}],
    # round 2 (coordinator, 2026-11-18 21:30): w5's two links in priority order, the seed-1 replicate of the passing
    # b256_15360_long first (does the pass belong to the recipe or to one draw), then b256_4375_long. Same runs as w5.
    "w5r": [{"run": "b256_15360_long_s1", "train": f"--size 256 --levels 8,8,8,6,5 {MIX} {REC} {LONG} --seed 1",
             "read": "--native256"},
            {"run": "b256_4375_long", "train": f"--size 256 --levels 7,5,5,5,5 {MIX} {REC} {LONG}",
             "read": "--native256"}],
    # round 2, item 3: the ONE combined 196-token attempt, submitted only if none of b224_15360_long, b224_64000_long
    # and b224_15360_wide_long passes: 64,000 codes, the wide model (117M) and the long schedule.
    "w7": [{"run": "b224_64000_wide_long",
            "train": f"--size 224 --levels 8,8,8,5,5,5 --widths 96,192,384,768 --attn-dim 768 {MIX} {REC} {LONG}"}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=sorted(CHAINS))
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rcs = []
    for link in CHAINS[a.chain]:
        cmd = [sys.executable, "-X", "faulthandler", "-u", "-m", "eo.experiments.segmentation.step1_B_rcp_run",
               "--run", link["run"], "--out-root", a.out_root]
        if link.get("read"):
            cmd.append(f"--read-args={link['read']}")
        cmd += ["--"] + shlex.split(link["train"])
        print(f"CHAIN {a.chain}: link {link['run']}: $ {shlex.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        print(f"CHAIN {a.chain}: link {link['run']} rc {rc}", flush=True)
        rcs.append(rc)
    sys.exit(0 if all(r == 0 for r in rcs) else 1)


if __name__ == "__main__":
    main()
