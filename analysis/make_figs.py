#!/usr/bin/env python3
"""Regenerate Figs 1-2 (ROUGE-1 vs Faith-ZS scatters) with plain system names,
manual label offsets (fixes the lead/bart and hybrid/led_ftraw collisions
reviewers flagged), and a regime legend. Data: cached main tables
(workspace/audit_drive/paper/table_main_{arxiv,pubmed}.csv) - the same
system means and 95% CIs plotted in the manuscript's current figures."""
import os, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def parse(v):
    m = re.match(r"([\d.\-]+) \[([\d.\-]+), ([\d.\-]+)\]", str(v).replace("\\textbf{","").replace("}",""))
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

NAMES = {
 "lead": "Lead-200", "extractive": "Extractive", "bart": "BART", "t5": "T5",
 "pegasus_arxiv": "PEGASUS", "pegasus_pubmed": "PEGASUS", "led_arxiv": "LED-large",
 "led_only": "Zero-shot LED", "hybrid": "Zero-shot hybrid",
 "hybrid_led_arxiv": "Zero-shot hybrid + LED",
 "led_ftraw_arxiv": "Fine-tuned raw", "led_ftraw_pubmed": "Fine-tuned raw",
 "hybrid_ft_arxiv": "Fine-tuned hybrid", "hybrid_ft_pubmed": "Fine-tuned hybrid",
}
STYLE = {"Heuristic": ("#1f77b4", "o"), "Off-the-shelf abstractive": ("#2ca02c", "s"),
         "Hybrid, zero-shot": ("#9467bd", "D"), "Fine-tuned LED-base (matched)": ("#d62728", "^")}
OFF_ARXIV = {"lead": (-4, 10), "bart": (8, -16), "t5": (8, 4), "extractive": (8, 4),
             "hybrid": (8, 4), "hybrid_led_arxiv": (8, 4), "led_arxiv": (8, 4),
             "pegasus_arxiv": (8, 4), "led_only": (8, 4),
             "hybrid_ft_arxiv": (-78, 6), "led_ftraw_arxiv": (8, -16)}
OFF_PUBMED = {"lead": (8, 4), "bart": (8, 4), "extractive": (8, 4), "t5": (8, 4),
              "led_arxiv": (8, 4), "hybrid_led_arxiv": (-118, 6), "hybrid": (-135, 18),
              "led_ftraw_pubmed": (12, -30), "led_only": (8, 4), "pegasus_pubmed": (8, 4),
              "hybrid_ft_pubmed": (8, 4)}
for ds, off, out in [("arxiv", OFF_ARXIV, "fig1_arxiv_revised.png"),
                     ("pubmed", OFF_PUBMED, "fig2_pubmed_revised.png")]:
    df = pd.read_csv(os.path.join(os.environ.get("BEYOND_ROUGE_DATA", "data"), f"table_main_{ds}.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    seen = set()
    for _, r in df.iterrows():
        x, xlo, xhi = parse(r["ROUGE-1"]); y, ylo, yhi = parse(r["Faith-ZS"])
        c, mk = STYLE[r["Group"]]
        ax.errorbar(x, y, xerr=[[x-xlo],[xhi-x]], yerr=[[y-ylo],[yhi-y]],
                    fmt=mk, color=c, ecolor=c, elinewidth=1, capsize=2, ms=7,
                    label=r["Group"] if r["Group"] not in seen else None)
        seen.add(r["Group"])
        dx, dy = off[r["System"]]
        ax.annotate(NAMES[r["System"]], (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9)
    ax.set_xlabel("ROUGE-1 (system mean, with 95% CI)")
    ax.set_ylabel("Faith-ZS (system mean, with 95% CI)")
    ax.set_xlim(29, 48); ax.set_ylim(25, 93)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), out), dpi=150)
    print("saved", out)
