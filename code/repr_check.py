#!/usr/bin/env python
"""Is the n=500 evaluation sample representative of the full test split? (review R2-W10)

Compares article length, abstract length and article/abstract compression of the seeded sample
(the SAME sample the study uses: seed 42, first n of the permutation) with the full test split:
mean/median/SD, standardised mean difference, two-sample KS test, and where the sample mean falls
inside the distribution of means of 5,000 random size-n subsamples (empirical two-sided p).

    python repr_check.py --dataset arxiv --n 500 --seed 42 --cache_dir DRIVE/data_cache --out DIR/representativeness_arxiv.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd


def compare(full: dict, samp: dict, n: int, rng, n_sub=5000):
    rows = []
    for k in full:
        f, s = np.asarray(full[k], float), np.asarray(samp[k], float)
        try:
            from scipy.stats import ks_2samp
            ks = ks_2samp(s, f)
            ksD, ksp = float(ks.statistic), float(ks.pvalue)
        except Exception:
            ksD = ksp = float("nan")
        sub_means = np.array([f[rng.choice(len(f), n, replace=False)].mean() for _ in range(n_sub)])
        emp_p = float(min(1.0, 2 * min((sub_means <= s.mean()).mean(), (sub_means >= s.mean()).mean())))
        rows.append(dict(variable=k, full_n=len(f), sample_n=len(s),
                         full_mean=f.mean(), sample_mean=s.mean(), full_median=np.median(f), sample_median=np.median(s),
                         full_sd=f.std(ddof=1), sample_sd=s.std(ddof=1),
                         smd=(s.mean() - f.mean()) / f.std(ddof=1), ks_D=ksD, ks_p=ksp,
                         sample_mean_vs_random_subsamples_p=emp_p,
                         subsample_mean_95pct_lo=np.percentile(sub_means, 2.5), subsample_mean_95pct_hi=np.percentile(sub_means, 97.5)))
    return pd.DataFrame(rows)


def lens(arts, abss):
    a = np.array([len(x.split()) for x in arts], float); b = np.array([len(x.split()) for x in abss], float)
    return {"article_words": a, "abstract_words": b, "compression": b / np.maximum(a, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv"); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import _table_from_parquet, PARQUET_CONFIG, load_subset
    from run_systems import DATASETS
    repo = DATASETS.get(a.dataset, a.dataset)
    samp = load_subset(repo, "test", a.n, a.seed, cache_dir=a.cache_dir)
    t = _table_from_parquet(repo, "test", PARQUET_CONFIG[repo], ["article", "abstract"])
    full = lens(t["article"].to_pylist(), t["abstract"].to_pylist())
    s = lens([d["article"] for d in samp], [d["abstract"] for d in samp])
    df = compare(full, s, a.n, np.random.default_rng(0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(3).to_string(index=False))
    print("Reading: |smd| < 0.1 and sample_mean_vs_random_subsamples_p > 0.05 => representative on that variable.")


if __name__ == "__main__":
    main()
