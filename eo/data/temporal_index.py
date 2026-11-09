# eo/data/temporal_index.py
"""The location and date index of the temporal stage (`.venv`).

A temporal sequence reads the four rows of one location and writes a date block for each. Neither
fact is in a token root: rows are linked only by `sample_id` (temporal_split.py), and dates sit only
in the metadata parquet, which is too big to open in a dataloader. So both are materialized once per
root, under <root>/temporal/:

    locations.npy        (L, 4) int64   the rows of each ssl4eos12 location, chronological
    dates.npy            (R, 2) uint8   year id and day-of-year bin of every row, both corpora
    temporal_build.json  input and array hashes, counts, the structural checks
    temporal_check.json  the token checks, written where the tokens are

A location's integer id is its line in locations.npy, sorted by the location string, so a draw keyed
on it names the same place on every machine. Major TOM rows are in dates.npy and not in locations.npy.

THE DATE IS THE ROW'S S2 DATE. Train's `S1_time` is a join artefact (date 0's date in all four rows),
and the true S1 date would mean reading all 96 tars. So a date block carries:

    year id   year - 2015, 0..11
    day bin   min((day of year - 1) // 5, 72), 0..72, so that days 361-366 share the last bin

These are positions in their own ranges, not vocabulary ids. ⚠ dates.npy is uint8: cast to int64
before adding a vocabulary offset, as with the uint16 tokens, or an offset near 87,589 wraps silently.
`TemporalIndex.date_of` does.

WHY TWO STEPS. The structural checks need the metadata and run at build. The token checks need the
tokens, and train's are on g49 only, so `check` runs there, on the arrays built here. Both hash what
they read: a regenerated tok_index.parquet would leave the rows valid and naming other scenes, which
no shape check sees.

WHAT THE TOKEN CHECK CAN SEE. DEM and Coords do not change between a location's dates, so a row from
another location shows up: it differs from the right one in a median of 195 of DEM's 196 tokens. It
stays hidden only if the two share both, which `StaticTokens.twins` counts, and the structural check
has no such blind spot, because it reads `sample_id`.

⚠ DEM IS NOT BIT-IDENTICAL ACROSS CARDS. Train's shards were encoded on a TITAN V and on A100s, and
5.0e-5 of DEM tokens differ between rows of the two (eo/experiments/temporal/dem_dates.py): 5,325
tokens in 539,680 pairs of rows, never more than 2 in a pair, each one FSQ digit moved by one level.
Pairs encoded on one card never differ (0 of 897,584). Hence STATIC_LIMITS, and not equality.

Runs in `.venv`, and in `mor-eo` on RCP: numpy and pandas, never terratorch.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from eo.data.eval_split import DEFAULT_METADATA_PATH, _sha256
from eo.data.temporal_split import N_DATES, load_temporal_split, location_table, row_dates, split_path
from eo.terramesh_tok.contract import tok_dir_name
from eo.terramesh_tok.train_split import atomic_json, atomic_npy, git_rev

INDEX_DIR = "temporal"
LOCATIONS_FILE = "locations.npy"
DATES_FILE = "dates.npy"
BUILD_FILE = "temporal_build.json"
CHECK_FILE = "temporal_check.json"
INDEX_NAME = "tok_index.parquet"

YEAR_FIRST = 2015
N_YEARS = 12                   # 2015-2026
DOY_BIN_DAYS = 5
N_DOY_BINS = 73

STATIC_MODALITIES = ("DEM", "Coords")                                   # unchanged across a location's dates
BLOCK_MODALITIES = ("Coords", "DEM", "LULC", "S2L2A", "S1GRD", "NDVI")  # every date must hold all of them
DATE_PAIRS = tuple(itertools.combinations(range(N_DATES), 2))
#: Tokens two rows of one location may differ by. DEM's cross-card flips reach 2 of 196 on train;
#: 4 is that with a margin, and any limit from 2 to 16 catches the same swapped rows. Coords come
#: from a CPU function of lat and lon, and may not differ at all.
STATIC_LIMITS = {"DEM": 4, "Coords": 0}
CONTROL_SEED = 20261107

#: Worked by hand, so that the rule is not checked against itself. 2020 is a leap year.
DATE_CASES: Tuple[Tuple[str, int, int], ...] = (
    ("2020-01-01", 5, 0), ("2020-01-05", 5, 0), ("2020-01-06", 5, 1),
    ("2020-02-29", 5, 11), ("2021-03-01", 6, 11),
    ("2019-12-31", 4, 72), ("2020-12-26", 5, 72), ("2020-12-31", 5, 72),
    ("2015-06-01", 0, 30), ("2026-12-31", 11, 72),
)
DATE_REFUSED = ("2014-12-31", "2027-01-01", None)


def index_dir(root_dir) -> Path:
    return Path(root_dir) / INDEX_DIR


def date_ids(s2_time) -> np.ndarray:
    """(R, 2) uint8: year id and day bin of each date. Raises on a missing or out-of-range date."""
    import pandas as pd

    t = pd.Series(pd.to_datetime(s2_time))
    if t.isna().any():
        raise RuntimeError(f"{int(t.isna().sum())} rows have no S2 date")
    year = t.dt.year.to_numpy(dtype=np.int64) - YEAR_FIRST
    if year.min() < 0 or year.max() >= N_YEARS:
        raise RuntimeError(f"S2 years {year.min() + YEAR_FIRST}..{year.max() + YEAR_FIRST} fall outside "
                           f"{YEAR_FIRST}..{YEAR_FIRST + N_YEARS - 1}")
    day = np.minimum((t.dt.dayofyear.to_numpy(dtype=np.int64) - 1) // DOY_BIN_DAYS, N_DOY_BINS - 1)
    return np.stack([year, day], axis=1).astype(np.uint8)


def date_rule_report(rule: Callable = date_ids) -> Dict:
    """The rule against DATE_CASES, and against the dates it must refuse."""
    wrong, accepted = [], []
    for day, year_id, day_bin in DATE_CASES:
        got = rule([day])[0]
        if (int(got[0]), int(got[1])) != (year_id, day_bin):
            wrong.append({"date": day, "got": [int(got[0]), int(got[1])], "want": [year_id, day_bin]})
    for day in DATE_REFUSED:
        try:
            rule([day])
        except RuntimeError:
            continue
        accepted.append(str(day))
    return {"cases": len(DATE_CASES), "wrong": wrong, "accepted_but_refused_by_contract": accepted,
            "passed": not wrong and not accepted}


def _wrong_rule(s2_time) -> np.ndarray:
    """The likeliest slip: day of year not moved to zero-based, and no last-bin clamp."""
    import pandas as pd

    t = pd.Series(pd.to_datetime(s2_time))
    return np.stack([t.dt.year.to_numpy() - YEAR_FIRST, t.dt.dayofyear.to_numpy() // DOY_BIN_DAYS], axis=1)


def structure_report(locations: np.ndarray, dates: np.ndarray, table) -> Dict:
    """Everything the metadata can say about the arrays, recomputed from it. Offences are counted,
    not raised, so a control can show which one caught it."""
    import pandas as pd

    n = len(table)
    ssl = (table.corpus == "ssl4eos12").to_numpy()
    flat = np.asarray(locations).ravel()
    out = {"locations": int(len(locations)), "rows": int(n),
           "wrong_shape": int(np.ndim(locations) != 2 or np.shape(locations)[1] != N_DATES
                              or np.shape(dates) != (n, 2)),
           "rows_out_of_range": int(((flat < 0) | (flat >= n)).sum())}
    if out["wrong_shape"] or out["rows_out_of_range"]:
        out["passed"] = False
        return out

    ids = table.sample_id[ssl].str.rsplit("_", n=1, expand=True)
    place = np.full(n, -1, dtype=np.int64)
    when = np.full(n, -1, dtype=np.int64)
    place[ssl] = pd.factorize(ids[0])[0]
    when[ssl] = ids[1].astype(np.int64).to_numpy()
    s2 = table.S2_time.to_numpy(dtype="datetime64[ns]")
    used = np.unique(flat)

    out.update({
        "rows_not_ssl4eos12": int((~ssl[flat]).sum()),
        "rows_repeated": int(flat.size - used.size),
        "ssl4eos12_rows_unused": int(ssl.sum() - ssl[used].sum()),
        "locations_mixed": int((place[locations] != place[locations[:, :1]]).any(axis=1).sum()),
        "locations_out_of_order": int((when[locations] != np.arange(N_DATES)).any(axis=1).sum()),
        "locations_not_increasing": int((np.diff(s2[locations], axis=1) <= np.timedelta64(0, "ns")).any(axis=1).sum()),
        "rows_with_wrong_date": int((np.asarray(dates) != date_ids(table.S2_time)).any(axis=1).sum()),
    })
    out["passed"] = not any(v for k, v in out.items() if k not in ("locations", "rows"))
    return out


def swap_rows(locations: np.ndarray, pairs: int, seed: int = CONTROL_SEED) -> np.ndarray:
    """A copy with one date's row exchanged between `pairs` pairs of locations."""
    rng = np.random.default_rng(seed)
    drawn = rng.permutation(len(locations))[: 2 * pairs]
    a, b = drawn[: pairs], drawn[pairs:]
    t = rng.integers(0, N_DATES, size=pairs)
    out = locations.copy()
    out[a, t], out[b, t] = locations[b, t], locations[a, t]
    return out


def structure_controls(locations: np.ndarray, dates: np.ndarray, table) -> Dict:
    """One deliberately broken copy per offence. `fired` means the intended offence was counted."""
    reversed_one = locations.copy()
    reversed_one[0] = reversed_one[0, ::-1]
    foreign = locations.copy()
    foreign[0, 0] = int(np.nonzero((table.corpus != "ssl4eos12").to_numpy())[0][0])
    repeated = locations.copy()
    repeated[1, 0] = repeated[0, 0]
    date_moved = dates.copy()
    date_moved[int(locations[0, 0]), 1] = (int(date_moved[int(locations[0, 0]), 1]) + 1) % N_DOY_BINS
    cases = {"rows_of_two_locations_swapped": (swap_rows(locations, 1), dates, "locations_mixed"),
             "one_location_reversed": (reversed_one, dates, "locations_not_increasing"),
             "a_majortom_row": (foreign, dates, "rows_not_ssl4eos12"),
             "a_row_used_twice": (repeated, dates, "rows_repeated"),
             "one_date_a_bin_off": (locations, date_moved, "rows_with_wrong_date")}
    out = {}
    for name, (broken, broken_dates, offence) in cases.items():
        rep = structure_report(broken, broken_dates, table)
        out[name] = {"offence": offence, "count": rep.get(offence, 0), "passed_anyway": rep["passed"],
                     "fired": bool(rep.get(offence, 0)) and not rep["passed"]}
    return out


def split_report(locations: np.ndarray, root_dir, metadata_path=None) -> Optional[Dict]:
    """On val, the index must hold exactly the locations of temporal_rows_v1.json. None on a root
    the split was not drawn on, which is told by the hash of its row order."""
    doc = json.loads(split_path().read_text())
    if doc["inputs"][INDEX_NAME] != _sha256(Path(root_dir) / INDEX_NAME):
        return None
    doc = load_temporal_split(root_dir, metadata_path=metadata_path)
    mine = {tuple(int(r) for r in line) for line in locations}
    theirs = {tuple(line) for side in ("dev", "test") for line in doc[side]["rows"]}
    return {"split_locations": len(theirs), "only_in_index": len(mine - theirs), "only_in_split": len(theirs - mine),
            "passed": mine == theirs}


def _counts(table, locations: np.ndarray, dates: np.ndarray) -> Dict:
    years, bins = {}, {}
    for corpus, g in table.groupby("corpus"):
        d = dates[g.index.to_numpy()]
        kept = np.bincount(d[:, 0], minlength=N_YEARS)
        years[str(corpus)] = {int(YEAR_FIRST + y): int(c) for y, c in enumerate(kept) if c}
        per_bin = np.bincount(d[:, 1], minlength=N_DOY_BINS)
        bins[str(corpus)] = {"fewest_rows": int(per_bin.min()), "most_rows": int(per_bin.max())}
    s2 = table.S2_time.to_numpy(dtype="datetime64[ns]")
    gap = np.diff(s2[locations], axis=1).astype("timedelta64[s]").astype(np.int64) / 86400
    return {"rows": int(len(table)), "locations": int(len(locations)),
            "rows_by_corpus": {str(k): int(v) for k, v in table.corpus.value_counts().sort_index().items()},
            "rows_by_year": years, "rows_per_day_bin": bins,
            "s2_gap_days": {"min": float(gap.min()), "median": float(np.median(gap)), "max": float(gap.max())}}


def build_index(root_dir, metadata_path=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """The two arrays and their record. Raises if a structural check fails or a control does not."""
    table = row_dates(root_dir, metadata_path)
    if not (table.row.to_numpy() == np.arange(len(table))).all():
        raise RuntimeError(f"{INDEX_NAME} does not number its rows 0..{len(table) - 1} in order")
    loc = location_table(root_dir, metadata_path, table=table)
    locations = loc[[f"row_{t}" for t in range(N_DATES)]].to_numpy(dtype=np.int64)
    dates = date_ids(table.S2_time)

    rule = date_rule_report()
    wrong_rule = date_rule_report(_wrong_rule)
    structure = structure_report(locations, dates, table)
    controls = structure_controls(locations, dates, table)
    split = split_report(locations, root_dir, metadata_path)
    if not rule["passed"] or wrong_rule["passed"]:
        raise RuntimeError(f"date rule: {rule}; the wrong rule passed: {wrong_rule['passed']}")
    if not structure["passed"]:
        raise RuntimeError(f"structural checks failed: {structure}")
    if not all(c["fired"] for c in controls.values()):
        raise RuntimeError(f"a structural control did not fire: {controls}")
    if split is not None and not split["passed"]:
        raise RuntimeError(f"the index and the committed split disagree: {split}")

    meta = Path(metadata_path or DEFAULT_METADATA_PATH)
    doc = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code": _code(),
           "date_rule": {"source": "S2_time", "year_first": YEAR_FIRST, "n_years": N_YEARS,
                         "doy_bin_days": DOY_BIN_DAYS, "n_doy_bins": N_DOY_BINS},
           "inputs": {INDEX_NAME: _sha256(Path(root_dir) / INDEX_NAME), "metadata": _sha256(meta),
                      "metadata_name": meta.name},
           "counts": _counts(table, locations, dates),
           "date_rule_check": rule, "date_rule_control": {"wrong": wrong_rule["wrong"], "fired": not wrong_rule["passed"]},
           "structure": structure, "structure_controls": controls, "split": split}
    return locations, dates, doc


def _code() -> str:
    me = _sha256(Path(__file__))[:12]
    return f"{git_rev()} temporal_index:{me}"


def write_index(root_dir, locations: np.ndarray, dates: np.ndarray, doc: Dict, force: bool = False) -> Path:
    """The arrays first and the record last, so a record always describes finished arrays. Refuses
    to overwrite: a location's id is its line, and draws are keyed on it."""
    out = index_dir(root_dir)
    if (out / BUILD_FILE).exists() and not force:
        raise FileExistsError(f"{out / BUILD_FILE} exists; a run keyed on its lines would read other places")
    atomic_npy(locations, out / LOCATIONS_FILE)
    atomic_npy(dates, out / DATES_FILE)
    doc = {**doc, "arrays": {LOCATIONS_FILE: _sha256(out / LOCATIONS_FILE), DATES_FILE: _sha256(out / DATES_FILE)}}
    atomic_json(doc, out / BUILD_FILE)
    return out


class StaticTokens:
    """The static modalities over the rows a location names, read once: the corpora are contiguous
    in the row order, so one slice serves every control, and a location is then an array lookup."""

    def __init__(self, root_dir, locations: np.ndarray, modalities: Sequence[str] = STATIC_MODALITIES):
        root = Path(root_dir)
        self.lo, hi = int(locations.min()), int(locations.max()) + 1
        self.blocks = {m: np.asarray(np.load(root / tok_dir_name(m) / "tokens.npy", mmap_mode="r")[self.lo: hi])
                       for m in modalities}

    def compare(self, locations: np.ndarray) -> Dict:
        """Over all pairs of a location's dates: how many locations differ, and how many by more
        than STATIC_LIMITS allows."""
        over = np.zeros(len(locations), dtype=bool)
        per = {}
        for mod, block in self.blocks.items():
            x = block[locations - self.lo]                  # (L, N_DATES, tokens)
            count = np.stack([(x[:, a] != x[:, b]).sum(axis=1) for a, b in DATE_PAIRS], axis=1)
            worst = count.max(axis=1)
            limit = STATIC_LIMITS[mod]
            per[mod] = {"limit": limit, "worst_pair": int(worst.max()), "locations_differing": int((worst > 0).sum()),
                        "locations_over_limit": int((worst > limit).sum()), "tokens_differing": int(count.sum()),
                        "tokens_compared": int(count.size * x.shape[2])}
            over |= worst > limit
        return {"per_modality": per, "locations_over_limit": int(over.sum()), "passed": not over.any()}

    def twins(self, locations: np.ndarray) -> int:
        """Locations whose static tokens equal another location's: a swap between two of them is
        invisible here, and only structure_report sees it."""
        first = locations[:, 0] - self.lo
        sig = np.concatenate([b[first] for b in self.blocks.values()], axis=1)
        _, inverse, counts = np.unique(sig, axis=0, return_inverse=True, return_counts=True)
        return int((counts[inverse.ravel()] > 1).sum())


def presence_report(root_dir, locations: np.ndarray, modalities: Sequence[str] = BLOCK_MODALITIES) -> Dict:
    """Every date must hold every block, or the sequence's grammar has a hole in it. Absent rows are
    zero-filled and zero is a real token, so the tokens alone cannot tell."""
    root, flat = Path(root_dir), locations.ravel()
    absent = {m: int(flat.size - np.load(root / tok_dir_name(m) / "present.npy", mmap_mode="r")[flat].sum())
              for m in modalities}
    return {"rows_absent": absent, "passed": not any(absent.values())}


def token_controls(tokens: StaticTokens, locations: np.ndarray) -> Dict:
    """Rows swapped between locations: one pair, 1% of the locations, then all of them."""
    n = len(locations)
    out = {}
    for name, pairs in (("one_pair", 1), ("one_percent", max(1, n // 200)), ("every_location", n // 2)):
        rep = tokens.compare(swap_rows(locations, pairs))
        out[name] = {"pairs": pairs, "locations_moved": 2 * pairs, "locations_over_limit": rep["locations_over_limit"],
                     "fired": not rep["passed"]}
    return out


def check_index(root_dir, modalities: Optional[Sequence[str]] = None, metadata_path=None) -> Dict:
    """The token checks on a built index, written to its temporal_check.json. `modalities` limits
    them to a part of the blocks, for a root that holds only some; the report is then not complete,
    and load_temporal_index refuses it."""
    report = token_report(root_dir, modalities, metadata_path)
    atomic_json(report, index_dir(root_dir) / CHECK_FILE)
    return report


def token_report(root_dir, modalities: Optional[Sequence[str]] = None, metadata_path=None) -> Dict:
    """check_index's report, written nowhere: a gate reruns the checks without rewriting the artifact
    (eo/scripts/verify_temporal.py, Y2)."""
    root = Path(root_dir)
    index = load_temporal_index(root, require_checked=False)
    build = index.build
    static = tuple(m for m in STATIC_MODALITIES if modalities is None or m in modalities)
    blocks = tuple(m for m in BLOCK_MODALITIES if modalities is None or m in modalities)
    metadata_equal = None
    if metadata_path is not None:
        metadata_equal = _sha256(Path(metadata_path)) == build["inputs"]["metadata"]
        if not metadata_equal:
            raise RuntimeError(f"{metadata_path} is not the metadata this index was built from")

    locations = index.locations
    tokens = StaticTokens(root, locations, static)
    report = {"checked": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code": _code(),
              "root": str(root), "inputs": {INDEX_NAME: build["inputs"][INDEX_NAME]}, "arrays": build["arrays"],
              "metadata_equal": metadata_equal, "complete": set(blocks) == set(BLOCK_MODALITIES),
              "presence": presence_report(root, locations, blocks), "static": tokens.compare(locations),
              "twins": tokens.twins(locations), "controls": token_controls(tokens, locations)}
    report["controls_fired"] = all(c["fired"] for c in report["controls"].values())
    report["passed"] = bool(report["presence"]["passed"] and report["static"]["passed"] and report["controls_fired"])
    return report


@dataclass(frozen=True)
class TemporalIndex:
    root: Path
    locations: np.ndarray           # (L, N_DATES) rows, chronological
    dates: np.ndarray               # (R, 2) uint8: year id, day bin
    build: Dict

    def __len__(self) -> int:
        return len(self.locations)

    def date_of(self, row: int) -> Tuple[int, int]:
        """Year id and day bin of a row, as Python ints: safe to add a vocabulary offset to."""
        year, day = self.dates[int(row)]
        return int(year), int(day)


def load_temporal_index(root_dir, require_checked: bool = True) -> TemporalIndex:
    """The index of a root, refused if it no longer describes that root. `require_checked` also
    asks for a complete token check of exactly these arrays."""
    root = Path(root_dir)
    d = index_dir(root)
    if not (d / BUILD_FILE).is_file():
        raise FileNotFoundError(f"no index at {d}: run eo/scripts/build_temporal_index.py build --root {root}")
    build = json.loads((d / BUILD_FILE).read_text())
    want, have = build["inputs"][INDEX_NAME], _sha256(root / INDEX_NAME)
    if want != have:
        raise RuntimeError(f"{d} was built against another {INDEX_NAME}: {want[:12]} vs {have[:12]}. "
                           f"Its row numbers would name other scenes.")
    for name, want in build["arrays"].items():
        have = _sha256(d / name)
        if want != have:
            raise RuntimeError(f"{d / name} is not the array {BUILD_FILE} records: {want[:12]} vs {have[:12]}")
    locations = np.load(d / LOCATIONS_FILE)
    dates = np.load(d / DATES_FILE, mmap_mode="r")
    counts = build["counts"]
    if locations.shape != (counts["locations"], N_DATES) or dates.shape != (counts["rows"], 2):
        raise RuntimeError(f"{d}: arrays {locations.shape}, {dates.shape} against counts {counts}")
    if require_checked:
        if not (d / CHECK_FILE).is_file():
            raise FileNotFoundError(f"{d / CHECK_FILE} missing: run eo/scripts/build_temporal_index.py check "
                                    f"where the tokens are")
        check = json.loads((d / CHECK_FILE).read_text())
        if not (check.get("passed") and check.get("complete")):
            raise RuntimeError(f"{d / CHECK_FILE}: passed {check.get('passed')}, complete {check.get('complete')}")
        if check["arrays"] != build["arrays"] or check["inputs"][INDEX_NAME] != build["inputs"][INDEX_NAME]:
            raise RuntimeError(f"{d / CHECK_FILE} checked other arrays than {BUILD_FILE} records")
    return TemporalIndex(root, locations, dates, build)
