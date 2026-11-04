#!/usr/bin/env python
"""Figures + trade-off tables from files already locally (review R2-W8; R1-W8). Generates NO model output.

  fig1_quality_vs_faithfulness_{ds}.png/.pdf  ROUGE-1 vs Faith-ZS for every system (95% CI bars) - the dissociation
  fig2_tradeoff_{ds}.png/.pdf                 ROUGE-1 (and Faith-ZS) vs latency; marker area = peak GPU memory
  fig3_learning_curve_{ds}.png/.pdf           hybrid-minus-raw difference vs fine-tuning scale (if learning_curve csv exists)
  tradeoff_table_{ds}.csv                     one row per system: quality, faithfulness, latency, memory (mean/SD over runs)
  efficiency_reduction_{ds}.csv               hybrid_ft vs led_ftraw: % latency / memory reduction with run-level SD
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse(c):
    m = CELL.match(str(c)); return tuple(float(x) for x in m.groups()) if m else (np.nan,) * 3


def load_quality(folder):
    df = pd.read_csv(Path(folder) / "summary.csv"); out = pd.DataFrame({"system": df["System"]})
    for col, k in [("ROUGE-1", "r1"), ("Faith-ZS", "faith")]:
        if col in df: out[[k, k + "_lo", k + "_hi"]] = [parse(x) for x in df[col]]
    return out


def regime(s):
    s = s.lower()
    if "repair" in s: return "verify-and-repair"
    if "ft" in s and "led_ftraw" not in s or "ftraw" in s: return "fine-tuned (matched pair)"
    if s.startswith("llm") or "qwen" in s: return "LLM"
    if s in ("lead", "extractive"): return "extractive"
    return "frozen / as released"


COL = {"extractive": "#7f7f7f", "frozen / as released": "#1f77b4", "fine-tuned (matched pair)": "#d62728", "LLM": "#9467bd", "verify-and-repair": "#2ca02c"}


def save(fig, base):
    fig.tight_layout(); fig.savefig(str(base) + ".png", dpi=200); fig.savefig(str(base) + ".pdf"); plt.close(fig); print("wrote", str(base) + ".png")


def fig1(q, ds, out):
    q = q.dropna(subset=["r1", "faith"]); fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for _, r in q.iterrows():
        c = COL[regime(r.system)]
        ax.errorbar(r.r1, r.faith, xerr=[[r.r1 - r.r1_lo], [r.r1_hi - r.r1]], yerr=[[r.faith - r.faith_lo], [r.faith_hi - r.faith]], fmt="o", color=c, ms=6, lw=1, capsize=2)
        ax.annotate(r.system, (r.r1, r.faith), textcoords="offset points", xytext=(5, 4), fontsize=7)
    for k, c in COL.items(): ax.plot([], [], "o", color=c, label=k)
    ax.set_xlabel("ROUGE-1 (x100)"); ax.set_ylabel("Faith-ZS (x100)"); ax.set_title(f"{ds}: content quality vs faithfulness (n=500, 95% CI)")
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=.3); save(fig, out / f"fig1_quality_vs_faithfulness_{ds}")


def fig2(t, ds, out):
    t = t.dropna(subset=["latency"]); 
    if t.empty: print("no efficiency data for", ds); return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (yk, yl) in zip(axs, [("r1", "ROUGE-1 (x100)"), ("faith", "Faith-ZS (x100)")]):
        for _, r in t.dropna(subset=[yk]).iterrows():
            ax.scatter(r.latency, r[yk], s=40 + 60 * (r.memory if not np.isnan(r.memory) else 1), color=COL[regime(r.system)], alpha=.8, edgecolor="k", lw=.5)
            ax.errorbar(r.latency, r[yk], xerr=r.latency_sd if not np.isnan(r.latency_sd) else 0, fmt="none", ecolor="k", lw=.6)
            ax.annotate(r.system, (r.latency, r[yk]), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("latency, s/document (log; mean +/- SD over 3 runs)"); ax.set_ylabel(yl); ax.grid(alpha=.3)
    for k, c in COL.items(): axs[0].scatter([], [], color=c, label=k)
    axs[0].legend(fontsize=7); fig.suptitle(f"{ds}: trade-off surface (marker area ~ peak GPU memory)"); save(fig, out / f"fig2_tradeoff_{ds}")


def fig3(lc, ds, out):
    d = lc[(lc.dataset == ds) & (lc.metric.isin(["r1", "faith_zs"]))] if "dataset" in lc else lc[lc.metric.isin(["r1", "faith_zs"])]
    if d.empty: return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m, lab in zip(axs, ["r1", "faith_zs"], ["ROUGE-1 diff (hybrid - raw, x100)", "Faith-ZS diff (hybrid - raw, x100)"]):
        s = d[d.metric == m].sort_values("n_train")
        ax.errorbar(s.n_train, s.diff_x100, yerr=[s.diff_x100 - s.lo_x100, s.hi_x100 - s.diff_x100], fmt="o-", capsize=3, color="#d62728")
        ax.axhline(0, color="k", lw=.8); ax.set_xlabel("fine-tuning pairs"); ax.set_ylabel(lab); ax.grid(alpha=.3); ax.set_xscale("log")
    fig.suptitle(f"{ds}: effect of pre-filtering vs training scale (95% paired-bootstrap CI)"); save(fig, out / f"fig3_learning_curve_{ds}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ds in ("arxiv", "pubmed"):
        qf = root / f"{ds}_n500" / "summary.csv"
        if not qf.exists(): print("SKIP", ds, "- no", qf); continue
        q = load_quality(root / f"{ds}_n500"); fig1(q, ds, out)
        ef = root / f"efficiency_{ds}_summary.csv"; t = q.copy()
        if ef.exists():
            e = pd.read_csv(ef).rename(columns={"mean_latency_s": "latency", "sd_latency_s": "latency_sd", "mean_peak_memory_GB": "memory", "sd_peak_memory_GB": "memory_sd"})
            t = q.merge(e[["system", "latency", "latency_sd", "memory", "memory_sd"]], on="system", how="outer")
            print(ds, "systems with quality but no efficiency row:", sorted(set(q.system) - set(e.system)))
            fig2(t, ds, out)
            for ha, hb in (("hybrid_ft", "led_ftraw"),):
                A, Bn = e[e.system.str.startswith(ha)], e[e.system.str.startswith(hb)]
                if len(A) == 1 and len(Bn) == 1:
                    A, Bn = A.iloc[0], Bn.iloc[0]
                    red = lambda x, y: 100 * (1 - x / y)
                    rows = [dict(metric="latency_s_per_doc", hybrid=A.latency, raw=Bn.latency, reduction_pct=red(A.latency, Bn.latency),
                                 runs_hybrid_sd=A.latency_sd, runs_raw_sd=Bn.latency_sd),
                            dict(metric="peak_memory_GB", hybrid=A.memory, raw=Bn.memory, reduction_pct=red(A.memory, Bn.memory),
                                 runs_hybrid_sd=A.memory_sd, runs_raw_sd=Bn.memory_sd)]
                    pd.DataFrame(rows).to_csv(out / f"efficiency_reduction_{ds}.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
        else:
            print("no", ef)
        t.to_csv(out / f"tradeoff_table_{ds}.csv", index=False)
        lcp = root / f"learning_curve_{ds}.csv"
        if lcp.exists(): fig3(pd.read_csv(lcp), ds, out)


if __name__ == "__main__":
    main()
