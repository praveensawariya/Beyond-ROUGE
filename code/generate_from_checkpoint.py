#!/usr/bin/env python
"""Generate n=500 predictions from a fine-tuned checkpoint, in the main-study format.

Use this for checkpoints that are NOT in run_systems.py's SYSTEMS table:
the Phase 3 scale-6k checkpoints and the Phase 4 FT-ablation variant
checkpoints. It mirrors run_systems.py exactly where it matters:
same seeded test sample (seed 42, first n of the shuffle), same
generation settings (beams 4, max_new 300, min_new 100, no-repeat-ngram 3,
length penalty 2.0, early stopping), same max_in (2048 for FT systems),
batch size 4 sorted by input length, amortised sec/doc, fsync-appended
resumable preds, and a .meta.json fingerprint.

Usage (run from the directory containing pipeline.py / run_systems.py,
i.e. base_code/, or set PYTHONPATH to it):
    python generate_from_checkpoint.py \
        --checkpoint models/led_ft_scale6k_arxiv_s7/model \
        --mode extracted --dataset arxiv --n 500 --seed 42 \
        --extract_cfg best_extract_arxiv.json \
        --cache_dir data_cache \
        --out scale6k_arxiv_n500 \
        --out_system hybrid_ft_scale6k_arxiv_s7

Then score with the original analyze.py:
    python analyze.py --results <out> --systems <out_system> --faithfulness
    python analyze.py --results <out> --systems <out_system> --faithfulness \
        --nli_model microsoft/deberta-large-mnli --tag nli2_
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to the checkpoint's model/ dir")
    ap.add_argument("--mode", choices=["extracted", "raw"], required=True)
    ap.add_argument("--dataset", default="arxiv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extract_cfg", default=None,
                    help="BASE (full) extraction config, e.g. best_extract_arxiv.json")
    ap.add_argument("--variant", choices=["full", "no_mmr", "no_lead", "no_ctx"], default="full",
                    help="Ablation applied on top of the base config (single-factor, asserted)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_system", required=True)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_in", type=int, default=2048)
    a = ap.parse_args()

    from data_utils import load_subset
    from hyb_gpu import gpu_report, require_gpu
    from pipeline import (Seq2Seq, clean_text, embed_sentences, extract,
                          free_gpu, make_gen_kwargs, set_seed, split_sentences)
    from run_systems import DATASETS, load_base_cfg, make_variants

    device = require_gpu()
    set_seed(a.seed)
    out = Path(a.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    ds_name = DATASETS.get(a.dataset, a.dataset)
    docs = load_subset(ds_name, "test", a.n, a.seed, cache_dir=a.cache_dir)
    idxs = [d["idx"] for d in docs]
    sents_of, refs = {}, {}
    for d in docs:
        sents = split_sentences(d["article"])
        sents_of[d["idx"]] = sents
        refs[d["idx"]] = {"idx": d["idx"], "reference": clean_text(d["abstract"]),
                          "article": " ".join(sents)}
    with open(out / "refs.jsonl", "w", encoding="utf-8") as f:
        for i in idxs:
            f.write(json.dumps(refs[i]) + "\n")

    variants = make_variants(load_base_cfg(a.extract_cfg))  # single-factor, asserted against the base
    cfg = variants[a.variant]

    pred_path = out / "preds" / f"{a.out_system}.jsonl"
    meta = {
        "system": a.out_system, "dataset": ds_name, "mode": a.mode,
        "checkpoint": str(a.checkpoint), "seed": a.seed, "n": a.n,
        "extract_cfg": str(a.extract_cfg) if a.extract_cfg else None,
        "variant": a.variant,
        "extract_full": vars(cfg) if a.mode == "extracted" else None,
        "gen": make_gen_kwargs(a.beams), "max_in_tokens": a.max_in,
        "batch_size": a.batch_size,
    }
    meta["fp"] = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
    (out / "preds" / f"{a.out_system}.meta.json").write_text(json.dumps(meta, indent=2))

    done = set()
    if pred_path.exists():
        done = {json.loads(l)["idx"] for l in pred_path.read_text().splitlines() if l.strip()}
    todo = [i for i in idxs if i not in done]
    print(f"[gen] {a.out_system}: {len(todo)} to generate, {len(done)} cached, checkpoint={a.checkpoint}")
    if not todo:
        print("Nothing to do - all predictions cached.")
        return

    emb_of = {}
    if a.mode == "extracted":
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        for i in idxs:
            emb_of[i] = embed_sentences(sents_of[i][:cfg.max_doc_sentences], sbert)
        del sbert
        free_gpu()

    def get_input(i):
        if a.mode == "raw":
            return " ".join(sents_of[i])
        s = sents_of[i][:cfg.max_doc_sentences]
        return extract(s, emb_of[i][:len(s)], cfg)["expanded_text"]

    model = Seq2Seq(str(a.checkpoint), "led", device)
    gen = make_gen_kwargs(a.beams)
    order = sorted(todo, key=lambda i: len(get_input(i).split()))
    for b in range(0, len(order), a.batch_size):
        chunk = order[b:b + a.batch_size]
        t0 = time.time()
        preds = model.summarize_batch([get_input(i) for i in chunk], a.max_in, gen)
        sec = (time.time() - t0) / len(chunk)
        with open(pred_path, "a", encoding="utf-8") as f:
            for i, p in zip(chunk, preds):
                f.write(json.dumps({"idx": i, "pred": p, "sec": round(sec, 3)}) + "\n")
                f.flush()
                os.fsync(f.fileno())
        if (b // a.batch_size) % 20 == 0:
            print(f"  {min(b + a.batch_size, len(order))}/{len(order)}")
    model.close()
    gpu_report("end of generation")
    print(f"Done. Next: python analyze.py --results {a.out} --systems {a.out_system} --faithfulness")


if __name__ == "__main__":
    main()
