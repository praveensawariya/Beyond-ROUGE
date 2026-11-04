#!/usr/bin/env python
"""Training-scale sensitivity of the pre-filtering effect (review R1-W5 / R2-W6).

Collects the paired hybrid_ft - led_ftraw difference at each fine-tuning scale from per-sample files
produced by analyze.py, with paired-bootstrap 95% CIs, so the paper can show whether the quality
penalty / faithfulness change shrinks, persists or reverses as training data grows.

    python learning_curve.py --dataset arxiv --out DIR/learning_curve_arxiv.csv \
        --point 3000:PS_A:hybrid_ft_arxiv:PS_B:led_ftraw_arxiv \
        --point 6000:PS_C:hybrid_ft_scale6k_arxiv_s7:PS_C:led_ftraw_scale6k_arxiv_s7 ...
Each --point = n_train:per_sample_csv_A:system_A:per_sample_csv_B:system_B  (A = hybrid, B = raw twin).
"""
import argparse
import numpy as np
import pandas as pd

METRICS = ["r1", "r2", "rl", "bertscore_f1", "faith_zs"]


def boot_diff(a, b, rng, B=10000):
    d = np.asarray(a, float) - np.asarray(b, float); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5]), min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))


def get(path, system):
    df = pd.read_csv(path)
    if "system" in df.columns: df = df[df.system == system]
    return df.drop_duplicates("idx").set_index("idx").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="")
    a = ap.parse_args(); rng = np.random.default_rng(0); rows = []
    for pt in a.point:
        n, pa, sa, pb, sb = pt.split(":")
        A, Bf = get(pa, sa), get(pb, sb); com = A.index.intersection(Bf.index)
        for m in METRICS:
            if m in A.columns and m in Bf.columns:
                d, lo, hi, p = boot_diff(A.loc[com, m], Bf.loc[com, m], rng)
                rows.append(dict(dataset=a.dataset, n_train=int(n), metric=m, n_docs=len(com), diff_x100=100 * d,
                                 lo_x100=100 * lo, hi_x100=100 * hi, p=p, ci_excludes_0=bool(lo > 0 or hi < 0)))
    df = pd.DataFrame(rows).sort_values(["metric", "n_train"]); df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))
    print("Verdict rule: same sign with CI excluding 0 at every scale => penalty is not an underfitting artefact.")


if __name__ == "__main__":
    main()
