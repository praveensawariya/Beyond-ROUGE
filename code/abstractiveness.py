#!/usr/bin/env python
"""Abstractiveness analysis WITH statistics (review R1-W9 / R2-W9).

Per summary: novel-unigram rate, novel-bigram rate, verbatim-sentence-copy rate.
  novel_n   = share of the summary's word n-grams that never occur in the source article
              (lower-cased, tokens = [a-z0-9]+)
  copy_sent = share of summary sentences (>=4 words) that occur verbatim (normalised) in the source
Reports mean [95% bootstrap CI] per system, and for each --pair A:B the paired document-level
bootstrap difference (A-B) for every metric, plus a LENGTH-ADJUSTED difference: the intercept of
  d_metric = alpha + beta * d_words      (d = A - B, per document)
i.e. the A-B difference at equal output length. This tests directly whether an abstractiveness
difference is an artefact of output length. Holm correction is applied over the metric family.

    python abstractiveness.py --results DIR --systems hybrid_ft_arxiv led_ftraw_arxiv \
        --pair hybrid_ft_arxiv:led_ftraw_arxiv --out DIR/abstractiveness
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

TOK = re.compile(r"[a-z0-9]+")
SPLIT = re.compile(r"(?<=[.!?])\s+")
B = 10000


def toks(t): return TOK.findall(t.lower())


def ngrams(w, n): return list(zip(*[w[i:] for i in range(n)])) if len(w) >= n else []


def novel_rate(summary_toks, source_set, n):
    g = ngrams(summary_toks, n)
    return float("nan") if not g else sum(x not in source_set for x in g) / len(g)


def copy_sent_rate(summary, source_norm):
    ss = [s for s in SPLIT.split(summary) if len(s.split()) >= 4]
    if not ss: return float("nan")
    return float(np.mean([" ".join(toks(s)) in source_norm for s in ss]))


def doc_metrics(summary, article):
    st, at = toks(summary), toks(article)
    uni, bi = set(at), set(ngrams(at, 2))
    return dict(words=len(summary.split()),
                novel_uni=float("nan") if not st else sum(w not in uni for w in st) / len(st),
                novel_bi=novel_rate(st, bi, 2),
                copy_sent=copy_sent_rate(summary, " ".join(at)))


def boot_ci(x, rng, B=B):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def paired(d, dw, rng, B=B):
    """d, dw: per-doc differences (A-B) in metric and words. Returns raw diff, adjusted diff, CIs, p-values."""
    ok = ~(np.isnan(d) | np.isnan(dw)); d, dw = d[ok], dw[ok]
    n = len(d)
    def adj(idx):
        x, y = dw[idx], d[idx]
        if np.ptp(x) == 0: return y.mean()
        return np.polyfit(x, y, 1)[1]                        # intercept = difference at equal length
    ib = rng.integers(0, n, (B, n))
    raw_b = d[ib].mean(1)
    adj_b = np.array([adj(r) for r in ib[:2000]])            # 2000 resamples is enough for the intercept CI
    p = lambda b: min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return dict(n=n, diff=d.mean(), lo=np.percentile(raw_b, 2.5), hi=np.percentile(raw_b, 97.5), p=p(raw_b),
                adj=adj(np.arange(n)), adj_lo=np.percentile(adj_b, 2.5), adj_hi=np.percentile(adj_b, 97.5), adj_p=p(adj_b))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); out[i] = min(1.0, run)
    return out


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--pair", action="append", default=[], help="A:B (repeatable)")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = Path(a.results); rng = np.random.default_rng(a.seed)
    refs = {r["idx"]: r for r in load_jsonl(res / "refs.jsonl")}
    rows = []
    for s in a.systems:
        for r in load_jsonl(res / "preds" / f"{s}.jsonl"):
            if r["idx"] in refs:
                rows.append(dict(system=s, idx=r["idx"], **doc_metrics(r["pred"], refs[r["idx"]]["article"])))
    df = pd.DataFrame(rows); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out) + "_per_sample.csv", index=False)
    mets = ["novel_uni", "novel_bi", "copy_sent", "words"]
    summ = []
    for s in a.systems:
        d = df[df.system == s]; row = {"system": s, "n": len(d)}
        for m in mets:
            mu, lo, hi = boot_ci(d[m], rng); row[m] = f"{mu*(1 if m=='words' else 100):.2f} [{lo*(1 if m=='words' else 100):.2f}, {hi*(1 if m=='words' else 100):.2f}]"
        summ.append(row)
    pd.DataFrame(summ).to_csv(str(out) + "_summary.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    prow = []
    for pr in a.pair:
        A, Bn = pr.split(":")
        da, db = df[df.system == A].set_index("idx"), df[df.system == Bn].set_index("idx")
        com = da.index.intersection(db.index)
        dw = (da.loc[com, "words"] - db.loc[com, "words"]).to_numpy(float)
        fam = []
        for m in ["novel_uni", "novel_bi", "copy_sent"]:
            r = paired((da.loc[com, m] - db.loc[com, m]).to_numpy(float), dw, rng); r["metric"] = m; fam.append(r)
        hp, hap = holm([r["p"] for r in fam]), holm([r["adj_p"] for r in fam])
        for r, p1, p2 in zip(fam, hp, hap):
            prow.append({"A": A, "B": Bn, "metric": r["metric"], "n": r["n"],
                         "diff_x100": f"{100*r['diff']:+.2f} [{100*r['lo']:+.2f}, {100*r['hi']:+.2f}]", "p_holm": round(p1, 4),
                         "length_adj_diff_x100": f"{100*r['adj']:+.2f} [{100*r['adj_lo']:+.2f}, {100*r['adj_hi']:+.2f}]", "adj_p_holm": round(p2, 4)})
    if prow:
        P = pd.DataFrame(prow); P.to_csv(str(out) + "_paired.csv", index=False); print(P.to_string(index=False))


if __name__ == "__main__":
    main()
