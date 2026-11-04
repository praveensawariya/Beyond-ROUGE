#!/usr/bin/env python
"""Aggregate repeated bench.py runs into mean / SD per system.

bench.py writes one row per system per run (mean sec/doc over its timed
documents, peak VRAM). The original study reported 3 runs of 50 documents;
run bench.py 3 times (e.g. --out efficiency_run1.csv, run2, run3) and then:

    python aggregate_efficiency.py --runs efficiency_run1.csv efficiency_run2.csv efficiency_run3.csv \
        --out efficiency_summary.csv

Output columns match final_runtime_statistics.csv (in the results directory): mean_latency_s, sd_latency_s, mean_peak_memory_GB,
sd_peak_memory_GB, plus the per-run values for the rebuttal appendix.
Report mean AND SD in the paper; do not report a single run as if it
were three.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames = []
    for k, path in enumerate(a.runs, start=1):
        df = pd.read_csv(path)
        df["run"] = k
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for system, grp in df.groupby("system"):
        rows.append({
            "system": system,
            "runs": len(grp),
            "documents_per_run": int(grp["actual_timed_docs"].iloc[0]) if "actual_timed_docs" in grp else None,
            "mean_latency_s": round(grp["sec_per_doc_mean"].mean(), 4),
            "sd_latency_s": round(grp["sec_per_doc_mean"].std(ddof=1), 4) if len(grp) > 1 else 0.0,
            "mean_peak_memory_GB": round(grp["peak_vram_GB"].mean(), 4),
            "sd_peak_memory_GB": round(grp["peak_vram_GB"].std(ddof=1), 4) if len(grp) > 1 else 0.0,
            "per_run_sec_per_doc": [round(x, 4) for x in grp["sec_per_doc_mean"]],
            "per_run_peak_GB": [round(x, 4) for x in grp["peak_vram_GB"]],
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
