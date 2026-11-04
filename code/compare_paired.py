#!/usr/bin/env python
"""Paired bootstrap comparison of two systems from analyze.py per-sample files.

Usage:
    python compare_paired.py --a variant_per_sample.csv --b full_per_sample.csv \
        --systems hyb_no_mmr_ft hybrid_ft --metrics r1 r2 rl bertscore_f1 faith_zs \
        --out paired_variant_vs_full.csv

The two files must share idx values. Differences are (a - b) x100, with
10,000-resample paired bootstrap 95% CIs -- the same convention as
paper_tables.py / length_bucket.py. This is the script for Phase 4
(variant FT vs full FT) and for Phase 3 (scale-6k hybrid vs scale-6k raw),
where the two systems live in different results folders and a single
analyze.py run cannot pair them.
"""
import argparse

import numpy as np
import pandas as pd

B = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="per_sample.csv for system A (first in the difference)")
    ap.add_argument("--b", required=True, help="per_sample.csv for system B")
    ap.add_argument("--systems", nargs=2, required=True, help="system names in A and B respectively")
    ap.add_argument("--metrics", nargs="+", default=["r1", "r2", "rl", "bertscore_f1", "faith_zs"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    da = pd.read_csv(a.a)
    db = pd.read_csv(a.b)
    # analyze.py per-sample column for the system name is 'system' in the
    # Gate 3 outputs; fall back to filtering by file if it is absent.
    for df, sysname in ((da, a.systems[0]), (db, a.systems[1])):
        if "system" in df.columns:
            df.query("system == @sysname", inplace=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for m in a.metrics:
        if m not in da.columns or m not in db.columns:
            print(f"[skip] metric {m} not in both files")
            continue
        x = da.set_index("idx")[m]
        y = db.set_index("idx")[m]
        common = x.index.intersection(y.index)
        d = (np.asarray(x.loc[common], float) - np.asarray(y.loc[common], float))
        d = d[~np.isnan(d)]
        boot = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"comparison": f"{a.systems[0]} - {a.systems[1]}", "metric": m,
                     "n": len(d), "diff_x100": round(100 * d.mean(), 2),
                     "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
