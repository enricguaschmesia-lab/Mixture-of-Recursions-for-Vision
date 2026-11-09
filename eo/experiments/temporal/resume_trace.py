#!/usr/bin/env python
"""A1 Step 8: does a run resumed mid-epoch on two ranks serve the micro-batches the uninterrupted run served?

Reads the trace that MOR_TEMPORAL_TRACE makes the training set write (eo/data/temporal_dataset.py): one line
per micro-batch served, with the pod (host), the rank, the draw epoch, the index, its members and a digest of
its five tensors. The run is killed after a checkpoint and Run:AI restarts it in a new pod, which resumes from
that checkpoint. So the steps between the checkpoint and the kill are served twice: by the uninterrupted run
(the first pod) and by the resumed one (the second).

Step g (1-based) is in epoch (g - 1) // spe, at step k = (g - 1) % spe of it. With a sequential sampler, batch 1
and accelerate's DDP sharding, rank r serves indices p = r, r + R, r + 2R, ... of its epoch, and step k takes
the A micro-batches at shard positions k*A .. k*A + A - 1: p = R * (k*A + a) + r.

Checks, each with a control that must fail (rule 2):
  C1 sharding    every micro-batch reached the rank DDP gives it (p % R)    | the ranks read swapped
  C2 the start   the resumed pod's first micro-batch on each rank is the first of step c + 1, and each rank's
                 indices then run on without a gap through its epochs, read from the serving order alone
                 (a new epoch is an index falling back by more than half an epoch), never from the recorded
                 epoch, which C3 checks                                   | the start one step later
  C3 draw epoch  the resumed pod drew every micro-batch at its step's epoch, global_step // spe
                 (TemporalEpochCallback)                                  | SetEpochCallback's int(round(state.epoch))
  C4 contents    every (epoch, index) both pods served has the same digest and members
                                                                          | the first pod one step before or
                                                                            after; the first pod one epoch back

    python eo/experiments/temporal/resume_trace.py TRACE_DIR --checkpoint 27 --steps-per-epoch 10 \
        [--accum 30] [--ranks 2] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ("time", "host", "rank", "pid", "epoch", "index", "stream", "n_dates", "n_members", "members_sha",
          "digest", "shape")


def read_trace(directory: Path) -> list:
    rows = []
    for f in sorted(directory.glob("trace_*.tsv")):
        for line in f.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != len(FIELDS):
                raise ValueError(f"{f.name}: {len(parts)} fields in {line[:80]!r}")
            r = dict(zip(FIELDS, parts))
            for k in ("rank", "pid", "epoch", "index", "n_dates", "n_members"):
                r[k] = int(r[k])
            r["time"] = float(r["time"])
            rows.append(r)
    return rows


def step_of(epoch: int, index: int, spe: int, accum: int, ranks: int) -> int:
    """The global step (1-based) that consumes micro-batch `index` of `epoch`."""
    return epoch * spe + (index // ranks) // accum + 1


def first_index(step: int, spe: int, accum: int, ranks: int, rank: int):
    """(epoch, index) of rank `rank`'s first micro-batch of global step `step`."""
    e, k = (step - 1) // spe, (step - 1) % spe
    return e, ranks * accum * k + rank


def gapless(seq) -> bool:
    return all(b == a + 1 for a, b in zip(seq, seq[1:]))


def served(seq, n: int, ranks: int, first_epoch: int) -> list:
    """A rank's lines in the order served (time), each with the epoch its place in that order gives it: a new
    epoch is recognised by the index falling back by more than half an epoch (workers reorder a few
    micro-batches, never half an epoch). The recorded epoch is not used: C3 compares it with this one."""
    out, it, top = [], first_epoch, None
    for r in sorted(seq, key=lambda x: x["time"]):
        q = r["index"] // ranks
        if top is not None and q < top - n // 2:
            it, top = it + 1, q
        else:
            top = q if top is None else max(top, q)
        out.append((it, r))
    return out


def analyse(rows, checkpoint: int, spe: int, accum: int, ranks: int) -> dict:
    hosts = sorted({r["host"] for r in rows}, key=lambda h: min(x["time"] for x in rows if x["host"] == h))
    if len(hosts) != 2:
        raise ValueError(f"expected the first pod and the resumed one, found hosts {hosts}")
    first, second = ([r for r in rows if r["host"] == h] for h in hosts)
    out = {"hosts": hosts, "micro_batches": [len(first), len(second)], "checkpoint": checkpoint,
           "steps_per_epoch": spe, "accum": accum, "ranks": ranks}
    n = spe * accum                                   # micro-batches per rank per epoch

    def by_rank(seg):
        d = defaultdict(list)
        for r in seg:
            d[r["rank"]].append(r)
        return d

    # Each pod's lines with the epoch their serving order gives them: the first pod from epoch 0, the
    # resumed one from the checkpoint's epoch.
    e_start = first_index(checkpoint + 1, spe, accum, ranks, 0)[0]
    lines1 = [x for seq in by_rank(first).values() for x in served(seq, n, ranks, 0)]
    lines2 = {rank: served(seq, n, ranks, e_start) for rank, seq in by_rank(second).items()}

    # C1: sharding.
    bad = [r for r in rows if r["index"] % ranks != r["rank"]]
    swapped = [r for r in rows if r["index"] % ranks != (ranks - 1 - r["rank"])]
    out["C1"] = {"passed": not bad, "offences": len(bad), "control_fired": len(swapped) > 0,
                 "control_offences": len(swapped)}

    # C2: where the resume starts, and no gap after it, from the serving order alone.
    starts, ctrl, firsts = {}, {}, {}
    for rank, seq in lines2.items():
        e0, p0 = first_index(checkpoint + 1, spe, accum, ranks, rank)
        e1, p1 = first_index(checkpoint + 2, spe, accum, ranks, rank)
        pos = sorted(e * n + r["index"] // ranks for e, r in seq)
        want, later = e0 * n + p0 // ranks, e1 * n + p1 // ranks
        starts[rank] = {"first_index_served": seq[0][1]["index"], "want_index": p0, "positions": [pos[0], pos[-1]],
                        "want_start": want, "gapless": gapless(pos), "served": len(pos),
                        "ok": pos[0] == want and gapless(pos)}
        ctrl[rank] = pos[0] == later and gapless(pos)
    for rank, seq in by_rank(first).items():
        pos = sorted(e * n + r["index"] // ranks for e, r in served(seq, n, ranks, 0))
        firsts[rank] = {"positions": [pos[0], pos[-1]], "gapless": gapless(pos), "from_step_1": pos[0] == 0}
    out["C2"] = {"passed": all(v["ok"] for v in starts.values()), "resumed": {str(k): v for k, v in starts.items()},
                 "first_pod": {str(k): v for k, v in firsts.items()}, "control_fired": not any(ctrl.values())}

    # C3: the epoch each micro-batch of the resumed pod was drawn at, against the epoch of the step that
    # serves it (its place in the serving order), and against SetEpochCallback's rounded epoch.
    rounded = int(round(checkpoint / spe))
    wrong, wrong_rounded, n_resumed_epoch = 0, 0, 0
    for seq in lines2.values():
        for e, r in seq:
            wrong += r["epoch"] != e
            if e == e_start:                  # the rest of the epoch the run resumed into
                n_resumed_epoch += 1
                wrong_rounded += r["epoch"] != rounded
    out["C3"] = {"passed": wrong == 0, "offences": wrong, "resumed_into_epoch": e_start, "rounded_epoch": rounded,
                 "micro_batches_in_that_epoch": n_resumed_epoch,
                 "control_fired": wrong_rounded > 0 and rounded != e_start, "control_offences": wrong_rounded}

    # C4: the same micro-batches, by content, keyed by the epoch and index of their place in each pod's order.
    a = {(e, r["index"]): r for e, r in lines1}
    b = {(e, r["index"]): r for seq in lines2.values() for e, r in seq}
    both = sorted(set(a) & set(b))
    diff = [k for k in both if (a[k]["digest"], a[k]["members_sha"]) != (b[k]["digest"], b[k]["members_sha"])]
    step = ranks * accum
    step_off = [(k, d) for k in b for d in (-step, step) if (k[0], k[1] + d) in a]
    off_same = [(k, d) for k, d in step_off if a[(k[0], k[1] + d)]["digest"] == b[k]["digest"]]
    other_epoch = [k for k in b if (k[0] - 1, k[1]) in a]
    other_same = [k for k in other_epoch if a[(k[0] - 1, k[1])]["digest"] == b[k]["digest"]]
    steps = sorted({step_of(e, p, spe, accum, ranks) for e, p in both})
    out["C4"] = {"passed": bool(both) and not diff, "served_by_both": len(both), "differing": len(diff),
                 "steps_served_by_both": [steps[0], steps[-1]] if steps else None,
                 "streams": {s: sum(1 for k in both if b[k]["stream"] == s) for s in ("temporal", "single")},
                 "control_one_step_off": {"pairs": len(step_off), "equal": len(off_same),
                                          "fired": bool(step_off) and not off_same},
                 "control_other_epoch": {"pairs": len(other_epoch), "equal": len(other_same),
                                         "fired": bool(other_epoch) and not other_same}}
    out["C4"]["control_fired"] = out["C4"]["control_one_step_off"]["fired"] and out["C4"]["control_other_epoch"]["fired"]
    out["passed"] = all(out[c]["passed"] and out[c]["control_fired"] for c in ("C1", "C2", "C3", "C4"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--checkpoint", type=int, required=True, help="the step the resumed pod restarted from")
    ap.add_argument("--steps-per-epoch", type=int, required=True)
    ap.add_argument("--accum", type=int, default=30)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = analyse(read_trace(args.trace), args.checkpoint, args.steps_per_epoch, args.accum, args.ranks)
    text = json.dumps(rep, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)
    for c in ("C1", "C2", "C3", "C4"):
        print(f"{c}: {'PASS' if rep[c]['passed'] else 'FAIL'}, control {'fired' if rep[c]['control_fired'] else 'DID NOT FIRE'}")
    raise SystemExit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
