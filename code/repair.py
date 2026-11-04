#!/usr/bin/env python
"""Verify-and-repair post-processing: drop generated sentences that the source does not support.
A sentence is kept only if (a) an NLI model finds support for it and (b) every number in it occurs in the paper.
IMPORTANT: use a DIFFERENT NLI model here than the one used in analyze.py (default below), otherwise the
faithfulness gain is circular (you would be optimising the metric you report)."""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
NUM = re.compile(r"\d+(?:\.\d+)?")


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--src_system", required=True)
    ap.add_argument("--out_system", required=True)
    ap.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--tau", type=float, default=0.5, help="min (P_entail - P_contra) of the best source window")
    ap.add_argument("--min_words", type=int, default=60, help="re-add best dropped sentences until the summary has this many words")
    a = ap.parse_args()
    from hyb_gpu import require_gpu
    device = require_gpu()
    from faithfulness import FaithfulnessScorer, sent_split
    sc = FaithfulnessScorer(nli_name=a.nli, device=device)
    res = Path(a.results)
    refs = {r["idx"]: r for r in jl(res / "refs.jsonl")}
    src = {r["idx"]: r for r in jl(res / "preds" / f"{a.src_system}.jsonl")}
    outp = res / "preds" / f"{a.out_system}.jsonl"
    done = {r["idx"] for r in jl(outp)} if outp.exists() else set()
    kept_tot = all_tot = 0
    with open(outp, "a", encoding="utf-8") as f:
        for n, (i, r) in enumerate(src.items()):
            if i in done or i not in refs: continue
            art, sents = refs[i]["article"], sent_split(r["pred"])
            srcnums = set(NUM.findall(art)); info = []
            for k, s in enumerate(sents):
                zs, _ = sc.score(i, art, s)
                ok = zs >= a.tau and all(x in srcnums for x in NUM.findall(s))
                info.append((k, s, zs, ok))
            keep = {k for k, _, _, ok in info if ok}
            words = sum(len(s.split()) for k, s, _, _ in info if k in keep)
            for k, s, zs, _ in sorted((x for x in info if x[0] not in keep), key=lambda x: -x[2]):
                if words >= a.min_words: break
                keep.add(k); words += len(s.split())
            text = " ".join(s for k, s, _, _ in info if k in keep)
            kept_tot += len(keep); all_tot += len(sents)
            f.write(json.dumps({"idx": i, "pred": text, "sec": r.get("sec", 0.0), "kept": len(keep), "total": len(sents)}) + "\n"); f.flush()
            if n % 25 == 0: print(f"  {n}/{len(src)}  kept {kept_tot}/{all_tot} sentences")
    print(f"done: kept {kept_tot}/{all_tot} sentences -> {outp}")


if __name__ == "__main__":
    main()
