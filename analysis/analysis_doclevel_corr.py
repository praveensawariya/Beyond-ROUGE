#!/usr/bin/env python3
"""Document-level ROUGE-1 vs Faith-ZS correlation (arXiv, n=500/system).
Source: workspace/drive_check_2026-10-21/arxiv_n500_per_sample.csv.
Excludes bigbird_arxiv (excluded system, not in the paper).
Vectorized Spearman + bootstrap CIs (10,000 resamples; pooled in chunks).
Cross-checks per-system means against manuscript Table 1 first."""
import os
import numpy as np, pandas as pd
# Per-sample arXiv table (500 docs x 11 systems) from the authors' materials;
# not in this repo. Point BEYOND_ROUGE_DATA at the folder holding it.
DATA = os.environ.get("BEYOND_ROUGE_DATA", "data")
F = os.path.join(DATA, "arxiv_n500_per_sample.csv")
OUT = os.environ.get("BEYOND_ROUGE_OUT", "doclevel_corr_arxiv.csv")
df = pd.read_csv(F)
df = df[df.system != "bigbird_arxiv"].copy()
chk = df.groupby("system").agg(r1=("r1","mean"), fz=("faith_zs","mean"))
print("Table 1 check (x100):"); print((chk*100).round(1).to_string())
def ranks(a):
    return pd.Series(a).rank().to_numpy()
def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    return np.corrcoef(rx, ry)[0,1]
def boot_spearman(x, y, B=10000, seed=42, chunk=2000):
    rx, ry = ranks(x), ranks(y)
    n = len(x); rng = np.random.default_rng(seed)
    ests = []
    done = 0
    while done < B:
        b = min(chunk, B-done)
        ix = rng.integers(0, n, (b, n))
        X, Y = rx[ix], ry[ix]
        Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
        ests.append((Xc*Yc).sum(1) / np.sqrt((Xc*Xc).sum(1)*(Yc*Yc).sum(1)))
        done += b
    return np.concatenate(ests)
rows = []
for sysname, g in sorted(df.groupby("system")):
    x, y = g["r1"].to_numpy(), g["faith_zs"].to_numpy()
    rho = spearman(x, y)
    b = boot_spearman(x, y)
    lo, hi = np.percentile(b, [2.5, 97.5])
    rows.append(dict(system=sysname, n=len(g), spearman=round(float(rho),3), ci_lo=round(float(lo),3), ci_hi=round(float(hi),3)))
    print(f"{sysname:18s} rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}]")
x, y = df["r1"].to_numpy(), df["faith_zs"].to_numpy()
rho = spearman(x, y)
b = boot_spearman(x, y, chunk=1000)
lo, hi = np.percentile(b, [2.5, 97.5])
rows.append(dict(system="POOLED_11_SYSTEMS", n=len(df), spearman=round(float(rho),3), ci_lo=round(float(lo),3), ci_hi=round(float(hi),3)))
print(f"POOLED rho={rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(df)}")
from scipy.stats import kendalltau
t1 = ["led_arxiv","pegasus_arxiv","led_ftraw_arxiv","extractive","hybrid_ft_arxiv","hybrid","lead","led_only"]
m = chk.loc[t1]
print(f"Kendall tau over 8 Table-1 systems: {kendalltau(m['r1'], m['fz']).statistic:.2f} (manuscript: -0.21)")
pd.DataFrame(rows).to_csv(OUT, index=False)
print("saved", OUT)
