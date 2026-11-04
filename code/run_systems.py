"""
Generate summaries for all systems on a fixed, seeded subset of a summarization dataset.
Resumable: every prediction is appended (and fsync'ed) to <out>/preds/<system>.jsonl and
finished samples are skipped on re-run.  Use a different --out per dataset.
Each system also gets <out>/preds/<system>.meta.json (fingerprint of every setting that changes
its output).  If you change such a setting later, old predictions are NOT silently mixed with new
ones: the run stops (--on_stale error) or archives them (--on_stale archive).
The GPU is REQUIRED by default (--device cuda): there is no silent CPU fallback.

    python run_systems.py --n 100 --systems lead extractive hybrid bart t5 led_only
    python run_systems.py --dataset pubmed --out results_pubmed --n 100 --systems lead hybrid bart
    python run_systems.py --extract_cfg results/best_extract.json --systems hybrid   # tuned config
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from data_utils import load_subset
from pipeline import (ExtractConfig, Seq2Seq, clean_text, embed_sentences, extract, free_gpu,
                      make_gen_kwargs, set_seed, split_sentences)

DATASETS = {"arxiv": "ccdv/arxiv-summarization", "pubmed": "ccdv/pubmed-summarization"}

MODELS = {   # key -> (HF id, kind)
    'led_ft_arxiv_s11': ('models/led_ft_arxiv_s11/model', 'led'),
    'led_ftraw_arxiv_s11': ('models/led_ftraw_arxiv_s11/model', 'led'),
    'led_ft_arxiv_s23': ('models/led_ft_arxiv_s23/model', 'led'),
    'led_ftraw_arxiv_s23': ('models/led_ftraw_arxiv_s23/model', 'led'),
    'led_ft_pubmed_s11': ('models/led_ft_pubmed_s11/model', 'led'),
    'led_ftraw_pubmed_s11': ('models/led_ftraw_pubmed_s11/model', 'led'),
    'led_ft_pubmed_s23': ('models/led_ft_pubmed_s23/model', 'led'),
    'led_ftraw_pubmed_s23': ('models/led_ftraw_pubmed_s23/model', 'led'),
    'pegasus_arxiv': ('google/pegasus-arxiv', 'pegasus'),
    'pegasus_pubmed': ('google/pegasus-pubmed', 'pegasus'),
    'led_ft_arxiv': ('models/led_ft_arxiv/model', 'led'),
    'led_ftraw_arxiv': ('models/led_ftraw_arxiv/model', 'led'),
    'led_ft_pubmed': ('models/led_ft_pubmed/model', 'led'),
    'led_ftraw_pubmed': ('models/led_ftraw_pubmed/model', 'led'),
    "led_base":  ("allenai/led-base-16384", "led"),
    "bart":      ("facebook/bart-large-cnn", "bart"),
    "t5":        ("google-t5/t5-base", "t5"),
    "led_arxiv": ("allenai/led-large-16384-arxiv", "led"),   # fine-tuned on arXiv (out-of-domain for PubMed)
}

# name -> (model, extractive variant | "raw" | None, text field, max input tokens)
SYSTEMS = {
    'hybrid_ft_arxiv_s11': ('led_ft_arxiv_s11', 'full', 'expanded_text', 2048),
    'led_ftraw_arxiv_s11': ('led_ftraw_arxiv_s11', 'raw', None, 'led_tokens'),
    'hybrid_ft_arxiv_s23': ('led_ft_arxiv_s23', 'full', 'expanded_text', 2048),
    'led_ftraw_arxiv_s23': ('led_ftraw_arxiv_s23', 'raw', None, 'led_tokens'),
    'hybrid_ft_pubmed_s11': ('led_ft_pubmed_s11', 'full', 'expanded_text', 2048),
    'led_ftraw_pubmed_s11': ('led_ftraw_pubmed_s11', 'raw', None, 'led_tokens'),
    'hybrid_ft_pubmed_s23': ('led_ft_pubmed_s23', 'full', 'expanded_text', 2048),
    'led_ftraw_pubmed_s23': ('led_ftraw_pubmed_s23', 'raw', None, 'led_tokens'),
    'pegasus_arxiv': ('pegasus_arxiv', 'raw', None, 1024),
    'pegasus_pubmed': ('pegasus_pubmed', 'raw', None, 1024),
    'hybrid_ft_arxiv': ('led_ft_arxiv', 'full', 'expanded_text', 2048),
    'led_ftraw_arxiv': ('led_ftraw_arxiv', 'raw', None, 'led_tokens'),
    'hybrid_ft_pubmed': ('led_ft_pubmed', 'full', 'expanded_text', 2048),
    'led_ftraw_pubmed': ('led_ftraw_pubmed', 'raw', None, 'led_tokens'),
    "lead":             (None, "raw", None, None),
    "extractive":       (None, "full", "core_text", None),
    "extractive_ctx":   (None, "full", "expanded_text", None),
    "hybrid":           ("led_base", "full", "expanded_text", 2048),
    "hyb_no_mmr":       ("led_base", "no_mmr", "expanded_text", 2048),
    "hyb_no_ctx":       ("led_base", "no_ctx", "expanded_text", 2048),
    "hyb_no_lead":      ("led_base", "no_lead", "expanded_text", 2048),
    "led_only":         ("led_base", "raw", None, "led_tokens"),
    "bart":             ("bart", "raw", None, 1024),
    "t5":               ("t5", "raw", None, 512),
    "led_arxiv":        ("led_arxiv", "raw", None, "led_tokens"),
    "hybrid_led_arxiv": ("led_arxiv", "full", "expanded_text", 2048),
}
LEAD_WORDS = 200


def make_variants(base: ExtractConfig) -> dict:
    """Extractive configurations; ablations are derived from the (possibly tuned) base.

    GATE 3 FIX (2026-10-21): every variant differs from "full" in EXACTLY ONE
    ExtractConfig field. The previous no_ctx also raised target_words 180->700
    and max_k 15->40 "so generator input length matches the full model" - that
    made it a three-variable change and its result unattributable to context
    removal. Here no_ctx changes ONLY window 1->0; the shorter generator input
    is the direct, intended consequence of removing context expansion, and the
    per-system Words/input lengths are reported by analyze.py so the size of
    that consequence is visible rather than compensated away.
    """
    variants = {
        "full":    base,
        "no_mmr":  replace(base, use_mmr=False),
        "no_lead": replace(base, lead_bias=0.0),
        "no_ctx":  replace(base, window=0),
    }
    assert_single_factor(variants)
    return variants


def assert_single_factor(variants: dict) -> None:
    """Fail loudly if any ablation variant differs from 'full' in >1 field."""
    from dataclasses import asdict as _asdict
    full = _asdict(variants["full"])
    for name, cfg in variants.items():
        if name == "full":
            continue
        diff = {k: (full[k], v) for k, v in _asdict(cfg).items() if v != full[k]}
        assert len(diff) == 1, f"variant {name} is not single-factor: {diff}"
        print(f"[gate3] variant {name}: single-factor diff {diff}")


def load_base_cfg(path):
    if not path:
        return ExtractConfig()
    d = json.loads(Path(path).read_text())
    d = d.get("best", d)
    return replace(ExtractConfig(), **{k: v for k, v in d.items() if k in ExtractConfig.__dataclass_fields__})


def tqdm_(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv", help="arxiv | pubmed | any HF dataset id")
    ap.add_argument("--split", default="test", help="use 'validation' for tuning, 'test' for reporting")
    ap.add_argument("--text_col", default="article")
    ap.add_argument("--summary_col", default="abstract")
    ap.add_argument("--n", type=int, default=100, help="number of articles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--systems", nargs="+", default=["lead", "extractive", "hybrid", "bart", "t5", "led_only"],
                    choices=list(SYSTEMS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--extract_cfg", default=None, help="JSON with tuned extractor values, e.g. configs/best_extract_arxiv.json (overrides the extractive defaults)")
    ap.add_argument("--beams", type=int, default=4, help="2 is ~2x faster on CPU")
    ap.add_argument("--led_tokens", type=int, default=4096,
                    help="input truncation for LED-only baselines (max 16384; RAM/time grow)")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="documents per GPU generate() call (1 = exact per-document latency; LED-only baselines always use 1)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cuda = GPU REQUIRED (aborts instead of silently running on CPU)")
    ap.add_argument("--on_stale", default="error", choices=["error", "archive"],
                    help="what to do when existing predictions were made with different settings")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cache_dir", default=None,
                    help="where the selected articles are cached (default: $HYB_DATA_CACHE, else no cache)")
    args = ap.parse_args()

    from hyb_gpu import gpu_report, require_gpu
    device = require_gpu() if args.device == "cuda" else "cpu"   # hard-fails if no usable GPU
    set_seed(args.seed)
    print(f"device = {device}")

    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    variants = make_variants(load_base_cfg(args.extract_cfg))

    ds_name = DATASETS.get(args.dataset, args.dataset)
    # seeded shuffle, first n (nested: a bigger --n reuses every cached prediction); only the needed
    # parquet shard is downloaded and the chosen articles are cached locally
    docs = load_subset(ds_name, args.split, args.n, args.seed, args.text_col, args.summary_col, args.cache_dir)
    idxs = [d["idx"] for d in docs]

    sents_of, refs = {}, {}
    for d in docs:
        i = d["idx"]
        sents = split_sentences(d["article"])
        sents_of[i] = sents
        refs[i] = {"idx": i, "reference": clean_text(d["abstract"]), "article": " ".join(sents)}
    with open(out / "refs.jsonl", "w", encoding="utf-8") as f:
        for i in idxs:
            f.write(json.dumps(refs[i]) + "\n")
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "dataset_id": ds_name, "device": device, "idxs": idxs,
         "variants": {k: asdict(v) for k, v in variants.items()}}, indent=2))

    def pred_path(name): return out / "preds" / f"{name}.jsonl"

    def system_fp(name):
        """Fingerprint of every setting that changes this system's output (N is excluded: nested)."""
        model, variant, _, max_in = SYSTEMS[name]
        d = {"system": name, "dataset": ds_name, "split": args.split, "seed": args.seed,
             "text_col": args.text_col, "summary_col": args.summary_col}
        if name == "lead":
            d["lead_words"] = LEAD_WORDS
        if variant not in (None, "raw"):
            d["extract"] = asdict(variants[variant])
            d["sbert"] = args.sbert
        if model:
            d["model"] = MODELS[model]
            d["gen"] = make_gen_kwargs(args.beams)
            d["max_in_tokens"] = args.led_tokens if max_in == "led_tokens" else max_in
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12], d

    def guard_stale(name):
        fp, desc = system_fp(name)
        p, mp = pred_path(name), out / "preds" / f"{name}.meta.json"
        has_preds = p.exists() and p.stat().st_size > 0
        if mp.exists():
            try:
                old = json.loads(mp.read_text()).get("fp")
            except json.JSONDecodeError:
                old = None
            if has_preds and old != fp:
                if args.on_stale == "archive":
                    dst = out / "preds" / "_stale" / f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
                    dst.parent.mkdir(exist_ok=True)
                    shutil.move(str(p), str(dst))
                    print(f"[stale] {name}: settings changed -> old predictions archived to {dst}")
                else:
                    raise SystemExit(
                        f"\n[stale] '{name}': predictions in {p} were produced with DIFFERENT settings than "
                        f"this run (e.g. a re-tuned extractor config, other beams/seed/model).\n"
                        f"Mixing them would corrupt the results. Either restore the old settings, or set "
                        f"\"on_stale\": \"archive\" in config.json to move the old predictions aside and "
                        f"regenerate this system.")
        elif has_preds:
            print(f"[meta] {name}: predictions from an earlier notebook version found - adopting current settings")
        mp.write_text(json.dumps({"fp": fp, "settings": desc}, indent=2))

    for name in args.systems:
        guard_stale(name)

    def done_ids(name):
        p = pred_path(name)
        if not p.exists():
            return set()
        recs, bad = [], 0
        raw = p.read_text(encoding="utf-8")
        for l in raw.splitlines():
            if l.strip():
                try:
                    recs.append(json.loads(l))
                except json.JSONDecodeError:         # last line cut off by an interrupted run
                    bad += 1
        if bad or (raw and not raw.endswith("\n")):   # repair so the next append starts on a clean line
            p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
            print(f"repaired {p.name}: dropped {bad} truncated line(s)")
        return {r["idx"] for r in recs}

    todo = {s: [i for i in idxs if i not in done_ids(s)] for s in args.systems}
    todo = {s: v for s, v in todo.items() if v}
    if not todo:
        print("Nothing to do - all predictions cached.")
        return

    emb_of = {}
    if any(SYSTEMS[s][1] not in ("raw", None) for s in todo):
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(args.sbert, device=device)
        print(f"[gpu] SBERT on {sbert.device}")
        print("embedding sentences ...")
        cap = variants["full"].max_doc_sentences
        for i in tqdm_(idxs):
            emb_of[i] = embed_sentences(sents_of[i][:cap], sbert)
        del sbert
        free_gpu()                     # embeddings are cached in RAM; give the VRAM back

    ext_cache = {}

    def get_input(system, i):
        _, variant, field, _ = SYSTEMS[system]
        if variant == "raw":
            return " ".join(sents_of[i])
        key = (i, variant)
        if key not in ext_cache:
            cfg = variants[variant]
            s = sents_of[i][:cfg.max_doc_sentences]
            ext_cache[key] = extract(s, emb_of[i][:len(s)], cfg)
        return ext_cache[key][field]

    def write(name, i, pred, sec):
        # flush + fsync: an interrupted run right after this call cannot lose the sample
        with open(pred_path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": i, "pred": pred, "sec": round(sec, 3)}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    for s in [s for s in todo if SYSTEMS[s][0] is None]:
        print(f"\n== {s} ({len(todo[s])} samples)")
        for i in todo[s]:
            t0 = time.time()
            pred = " ".join(get_input(s, i).split()[:LEAD_WORDS]) if s == "lead" else get_input(s, i)
            write(s, i, pred, time.time() - t0)

    gen = make_gen_kwargs(args.beams)
    for mkey in MODELS:
        group = [s for s in todo if SYSTEMS[s][0] == mkey]
        if not group:
            continue
        hf_name, kind = MODELS[mkey]
        print(f"\nloading {hf_name} ...")
        model = Seq2Seq(hf_name, kind, device)
        for s in group:
            max_in = SYSTEMS[s][3]
            max_in = args.led_tokens if max_in == "led_tokens" else max_in
            print(f"\n== {s} ({len(todo[s])} samples, max_in={max_in} tokens)")
            bs = 1 if SYSTEMS[s][3] == "led_tokens" else max(1, args.batch_size)
            order = sorted(todo[s], key=lambda i: len(get_input(s, i).split()))   # similar lengths -> less padding
            for b in tqdm_(range(0, len(order), bs)):
                chunk = order[b:b + bs]
                t0 = time.time()
                preds = model.summarize_batch([get_input(s, i) for i in chunk], max_in, gen)
                sec = (time.time() - t0) / len(chunk)        # amortised seconds per document
                for i, p in zip(chunk, preds):
                    write(s, i, p, sec)
        model.close()                                         # frees this model's VRAM before the next loads

    gpu_report("end of run")
    free_gpu()
    print(f"\nDone. Next: python analyze.py --results {args.out} --systems " + " ".join(args.systems))


if __name__ == "__main__":
    main()
