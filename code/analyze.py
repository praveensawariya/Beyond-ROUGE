"""
Score all cached predictions: ROUGE-1/2/L, BERTScore-F1, length/compression,
unsupported-number rate, latency; 95% bootstrap CIs and paired bootstrap
significance tests against a chosen baseline system.

    python analyze.py --systems lead extractive hybrid bart t5 led_only --baseline hybrid

Restart-safe: BERTScore and faithfulness are cached per (system, article, prediction-hash) in
<results>/score_cache/, written as they are computed.  After an interrupted run the run continues
with the missing items only; a changed prediction is re-scored automatically (its hash differs).
The GPU is REQUIRED for BERTScore / faithfulness (--device cuda) - no silent CPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import free_gpu

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def load_jsonl(p):
    out = []
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:          # truncated last line after an interrupted run
                pass
    return out


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


class ScoreCache:
    """Append-only JSONL cache: key = (article idx, hash of the prediction text)."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = {}
        if self.path.exists():
            for r in load_jsonl(self.path):
                self.d[(r["idx"], r["h"])] = r

    def get(self, idx, h):
        return self.d.get((idx, h))

    def put_many(self, recs):
        with open(self.path, "a", encoding="utf-8") as f:
            for r in recs:
                self.d[(r["idx"], r["h"])] = r
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())                 # survive a disconnect right after this call


def unsupported_numbers(summary: str, article: str):
    """Share of numbers in the summary that never occur in the source.
    Crude hallucination proxy - NOT a substitute for a faithfulness metric."""
    nums = NUM_RE.findall(summary)
    if not nums:
        return np.nan
    src = set(NUM_RE.findall(article))
    return sum(n not in src for n in nums) / len(nums)


def bootstrap_ci(x, B=10000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def paired_bootstrap(a, b, B=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return d.mean(), *np.percentile(means, [2.5, 97.5]), p


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def tex_table(df, caption, label):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    rows = [" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\small",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
                      " & ".join(esc(c) for c in cols) + r" \\", r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--baseline", default="hybrid", help="system tested against all others")
    ap.add_argument("--bert_model", default="roberta-large",
                    help="e.g. distilbert-base-uncased for speed/low RAM")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cuda = GPU REQUIRED for BERTScore/faithfulness (no silent CPU fallback)")
    ap.add_argument("--title", default=None, help="LaTeX table caption (default: derived from --results)")
    ap.add_argument("--faithfulness", action="store_true",
                    help="add NLI faithfulness (faith_zs, supported_rate); needs a model download")
    ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--tag", default="", help="prefix for output files, e.g. ablation_")
    args = ap.parse_args()

    use_neural = (not args.no_bertscore) or args.faithfulness
    if use_neural:
        from hyb_gpu import gpu_report, require_gpu
        dev = require_gpu() if args.device == "cuda" else "cpu"
    cache_dir = Path(args.results) / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    res = Path(args.results)
    refs = {r["idx"]: r for r in load_jsonl(res / "refs.jsonl")}
    preds = {s: {r["idx"]: r for r in load_jsonl(res / "preds" / f"{s}.jsonl")} for s in args.systems}

    common = sorted(set.intersection(*[set(p) for p in preds.values()]) & set(refs))
    print(f"{len(common)} samples common to all systems: {args.systems}")
    if not common:
        raise SystemExit("No overlapping samples - run run_systems.py first.")

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    for s in args.systems:
        for i in common:
            p, r, a = preds[s][i]["pred"], refs[i]["reference"], refs[i]["article"]
            sc = scorer.score(r, p)
            aw = max(1, len(a.split()))
            rows.append(dict(system=s, idx=i,
                             r1=sc["rouge1"].fmeasure, r2=sc["rouge2"].fmeasure, rl=sc["rougeL"].fmeasure,
                             words=len(p.split()), compression=len(p.split()) / aw,
                             unsup_num=unsupported_numbers(p, a), sec=preds[s][i]["sec"]))
    df = pd.DataFrame(rows)

    meta = {"rouge_score": "rouge1/rouge2/rougeL F1, Porter stemmer", "n_articles": len(common)}
    if not args.no_bertscore:
        from bert_score import BERTScorer
        print(f"BERTScore ({args.bert_model}) on {dev} ...")
        scorer_b, bf = None, []
        for s in args.systems:
            cache = ScoreCache(cache_dir / f"bert__{slug(args.bert_model)}__{s}.jsonl")
            hs = {i: md5(preds[s][i]["pred"] or " ") for i in common}
            miss = [i for i in common if cache.get(i, hs[i]) is None]
            if miss:
                if scorer_b is None:
                    scorer_b = BERTScorer(model_type=args.bert_model, lang="en", batch_size=16, device=dev)
                    meta["bertscore_hash"] = getattr(scorer_b, "hash", None)
                    gpu_report("BERTScore model loaded")
                for b in range(0, len(miss), 32):                 # save after every chunk
                    ch = miss[b:b + 32]
                    _, _, f1 = scorer_b.score([preds[s][i]["pred"] or " " for i in ch],
                                              [refs[i]["reference"] for i in ch])
                    cache.put_many([{"idx": i, "h": hs[i], "f1": float(x)} for i, x in zip(ch, f1.tolist())])
            print(f"  BERTScore {s}: {len(miss)} newly scored, {len(common) - len(miss)} from cache")
            bf += [cache.get(i, hs[i])["f1"] for i in common]
        df["bertscore_f1"] = bf
        meta["bertscore_model"] = args.bert_model
        del scorer_b
        free_gpu()                                   # drop roberta-large before the NLI model loads

    if args.faithfulness:
        from faithfulness import FaithfulnessScorer
        print(f"Faithfulness ({args.nli_model}) on {dev} ...")
        fs, zs, sup = None, [], []
        for s in args.systems:
            cache = ScoreCache(cache_dir / f"faith__{slug(args.nli_model)}__{s}.jsonl")
            hs = {i: md5(preds[s][i]["pred"] or " ") for i in common}
            miss = [i for i in common if cache.get(i, hs[i]) is None]
            if miss and fs is None:
                fs = FaithfulnessScorer(args.nli_model, device=dev, batch_size=32 if dev == "cuda" else 16)
                gpu_report("NLI + SBERT loaded")
            for n_done, i in enumerate(miss, 1):                  # save after every article
                a, b = fs.score(i, refs[i]["article"], preds[s][i]["pred"])
                cache.put_many([{"idx": i, "h": hs[i], "zs": a, "sup": b}])
                if n_done % 25 == 0:
                    print(f"    {s}: {n_done}/{len(miss)}")
            print(f"  {s} done ({len(miss)} newly scored, {len(common) - len(miss)} from cache)")
            zs += [cache.get(i, hs[i])["zs"] for i in common]
            sup += [cache.get(i, hs[i])["sup"] for i in common]
        df["faith_zs"], df["supported_rate"] = zs, sup
        meta["faithfulness_model"] = args.nli_model
        del fs
        free_gpu()

    mp = res / f"{args.tag}metrics_meta.json"
    if mp.exists():                              # keep e.g. the BERTScore hash from an earlier (non-cached) run
        try:
            meta = {**json.loads(mp.read_text()), **{k: v for k, v in meta.items() if v is not None}}
        except json.JSONDecodeError:
            pass
    mp.write_text(json.dumps(meta, indent=2))
    df.to_csv(res / f"{args.tag}per_sample.csv", index=False)

    metrics = [("r1", "ROUGE-1"), ("r2", "ROUGE-2"), ("rl", "ROUGE-L")]
    if "bertscore_f1" in df:
        metrics.append(("bertscore_f1", "BERTScore-F1"))
    if "faith_zs" in df:
        metrics += [("faith_zs", "Faith-ZS"), ("supported_rate", "Supported-sent")]

    summ = []
    for s in args.systems:
        d = df[df.system == s]
        row = {"System": s}
        for k, name in metrics:
            m, lo, hi = bootstrap_ci(d[k], args.boot)
            row[name] = f"{100*m:.2f} [{100*lo:.2f}, {100*hi:.2f}]"
        row["Words"] = f"{d.words.mean():.0f}"
        row["Compr."] = f"{d.compression.mean():.3f}"
        row["Unsup.num %"] = f"{100*np.nanmean(d.unsup_num):.1f}"
        row["Sec/doc"] = f"{d.sec.mean():.1f}"
        summ.append(row)
    summ = pd.DataFrame(summ)

    sig = []
    if args.baseline in args.systems:
        base = df[df.system == args.baseline].set_index("idx")
        for s in args.systems:
            if s == args.baseline:
                continue
            o = df[df.system == s].set_index("idx").loc[base.index]
            for k, name in metrics:
                md, lo, hi, p = paired_bootstrap(base[k], o[k], args.boot)
                sig.append({"Baseline": args.baseline, "vs": s, "Metric": name,
                            "Mean diff (x100)": f"{100*md:+.2f}",
                            "95% CI": f"[{100*lo:+.2f}, {100*hi:+.2f}]", "p": f"{p:.4f}"})
    sig = pd.DataFrame(sig)

    summ.to_csv(res / f"{args.tag}summary.csv", index=False)
    if len(sig):
        sig.to_csv(res / f"{args.tag}significance.csv", index=False)
    (res / f"{args.tag}summary.md").write_text(
        f"n = {len(common)} articles. Mean [95% bootstrap CI], scores x100.\n\n"
        + md_table(summ) + ("\n\nPaired bootstrap (positive diff = baseline better):\n\n" + md_table(sig) if len(sig) else ""),
        encoding="utf-8")
    (res / f"{args.tag}summary.tex").write_text(
        tex_table(summ.drop(columns=["Unsup.num %", "Sec/doc"]),
                  args.title or f"Results on {len(common)} {res.name} test articles (mean [95\\% CI]).",
                  f"tab:{res.name}{args.tag.rstrip('_') and '_' + args.tag.rstrip('_')}"),
        encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\n" + summ.to_string(index=False))
    if len(sig):
        print("\n" + sig.to_string(index=False))
    print(f"\nSaved to {res}/ ({args.tag}summary.md/.csv/.tex, {args.tag}significance.csv, {args.tag}per_sample.csv)")


if __name__ == "__main__":
    main()
