#!/usr/bin/env python3
"""Copy-adjusted Faith-ZS for the fine-tuned pair, arXiv only (exploratory).
Joins per-document Faith-ZS (arxiv_n500_per_sample.csv, Drive check 2026-10-21)
with per-document novel-bigram rates recomputed with the repo's
abstractiveness.py doc_metrics on preds_verify + test_cache_arxiv.jsonl.
Reports the paired hybrid_ft - led_ftraw Faith-ZS difference:
 raw; adjusted for output-length difference; adjusted for novel-bigram-rate
 difference; adjusted for both (intercept of per-document OLS); and within
 terciles of the abstractiveness gap (the per-document novel-bigram-rate
 difference, hybrid_ft - led_ftraw). 95% paired-bootstrap CIs
 (10,000 resamples; 2,000 for regression intercepts, as in abstractiveness.py).
Provenance note: novel-bigram system means on this test-cache copy are
 ~1.5 points above the manuscript's Sect. 5.5 values (different article copy);
 the paired difference reproduces Table A1 (+5.18 vs +5.16)."""
import os
import numpy as np, pandas as pd
# This folder (holds the per-document novel-bigram CSVs); per-sample arXiv
# table from the authors' materials (not in this repo).
R = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BEYOND_ROUGE_DATA", "data")
ps = pd.read_csv(os.path.join(DATA, "arxiv_n500_per_sample.csv"))
fz = ps[ps.system.isin(["hybrid_ft_arxiv","led_ftraw_arxiv"])].pivot(index="idx", columns="system", values="faith_zs")
wd = ps[ps.system.isin(["hybrid_ft_arxiv","led_ftraw_arxiv"])].pivot(index="idx", columns="system", values="words")
nb_h = pd.read_csv(f"{R}/abstractiveness_perdoc_arxiv_hybrid_ft.csv").set_index("idx")["novel_bi"]*100
nb_r = pd.read_csv(f"{R}/abstractiveness_perdoc_arxiv_led_ftraw.csv").set_index("idx")["novel_bi"]*100
d = pd.DataFrame({
    "d_fz": (fz["hybrid_ft_arxiv"] - fz["led_ftraw_arxiv"])*100,
    "d_nb": nb_h - nb_r,
    "d_w": wd["hybrid_ft_arxiv"] - wd["led_ftraw_arxiv"],
    "nb_mean": (nb_h + nb_r)/2,
})
print("n =", len(d), "| raw d_fz mean %.2f" % d["d_fz"].mean(),
      "| d_nb mean %.2f" % d["d_nb"].mean(), "| d_w mean %.2f" % d["d_w"].mean())
rng = np.random.default_rng(42)
B = 10000
def boot_stat(fn, n_boot=B):
    n = len(d)
    ib = rng.integers(0, n, (n_boot, n))
    return np.array([fn(i) for i in ib])
def ci(b):
    return np.percentile(b, [2.5, 97.5])
raw_b = boot_stat(lambda i: d["d_fz"].to_numpy()[i].mean())
print(f"raw d_fz: {d['d_fz'].mean():+.2f} [{ci(raw_b)[0]:+.2f}, {ci(raw_b)[1]:+.2f}]")
def intercept(y, X, idx):
    Xb = np.column_stack([np.ones(len(idx))] + [X[j][idx] for j in range(X.shape[1])]) if X.ndim==2 else None
    return np.polyfit(np.column_stack([X[j][idx] for j in range(X.shape[1])]) if False else None, y[idx], 1)[1] if False else None
# use OLS via lstsq for intercepts (cleaner than polyfit for multivariate)
def ols_intercept(y, Xs, idx):
    A = np.column_stack([np.ones(len(idx))] + [x[idx] for x in Xs])
    return np.linalg.lstsq(A, y[idx], rcond=None)[0][0]
y = d["d_fz"].to_numpy()
for label, Xs in [("length-adjusted (d_words)", [d["d_w"].to_numpy()]),
                  ("abstractiveness-adjusted (d_novel_bigram)", [d["d_nb"].to_numpy()]),
                  ("both-adjusted", [d["d_w"].to_numpy(), d["d_nb"].to_numpy()])]:
    est = ols_intercept(y, Xs, np.arange(len(d)))
    b = boot_stat(lambda i, Xs=Xs: ols_intercept(y, Xs, i), n_boot=2000)
    print(f"{label}: {est:+.2f} [{ci(b)[0]:+.2f}, {ci(b)[1]:+.2f}]")
# terciles of the abstractiveness gap (hybrid_ft - led_ftraw novel-bigram rate)
# Fresh seed for this block: the intervals reported in the manuscript (Sect. 5.5c)
# were computed with default_rng(42) at this step.
rng = np.random.default_rng(42)
d["terc"] = pd.qcut(d["d_nb"], 3, labels=["small","mid","large"])
for t, g in d.groupby("terc", observed=True):
    n = len(g)
    ib = rng.integers(0, n, (B, n))
    b = g["d_fz"].to_numpy()[ib].mean(1)
    print(f"tercile {t}: d_nb mean {g['d_nb'].mean():+.2f}  d_fz {g['d_fz'].mean():+.2f} [{ci(b)[0]:+.2f}, {ci(b)[1]:+.2f}]  n={n}")
d.to_csv(f"{R}/copy_adjusted_arxiv.csv", index=False)
print("saved copy_adjusted_arxiv.csv")
