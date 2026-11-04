#!/usr/bin/env python
"""Fine-tune LED-base so the abstractive stage is trained for its actual input.
   --mode extracted : input = extractor's expanded_text   (-> the 'hybrid_ft' system)
   --mode raw       : input = first --max_in tokens of the paper (-> matched control 'led_ftraw')
Both use the same base model, data, steps and optimiser, so any difference comes from the input."""
import argparse, json, math, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_train_subset(repo, split, n, seed, cache_dir=None, config="section"):
    """Memory-safe sampler for the (multi-GB) train split: read ONE parquet shard at a time and sample rows from each."""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    cache = Path(cache_dir) / f"{repo.replace('/', '__')}__{split}__seed{seed}__n{n}__shardsample.jsonl" if cache_dir else None
    if cache is not None and cache.exists():
        docs = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(docs) >= n:
            print(f"[data] {n} {split} docs from cache {cache.name}"); return docs[:n]
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"{config}/{split}-") and f.endswith(".parquet"))
    if not files: raise FileNotFoundError(f"no parquet shards for {repo}:{config}/{split}")
    rng, per, docs = np.random.default_rng(seed), -(-n // len(files)), []
    for k, f in enumerate(files):
        t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=["article", "abstract"])
        t = t.cast(pa.schema([("article", pa.large_string()), ("abstract", pa.large_string())]))
        pick = sorted(rng.choice(t.num_rows, min(per, t.num_rows), replace=False).tolist())
        sub = t.take(pick)
        docs += [{"idx": k * 10**7 + i, "article": a, "abstract": b}
                 for i, a, b in zip(pick, sub["article"].to_pylist(), sub["abstract"].to_pylist())]
        del t, sub
        print(f"  shard {k+1}/{len(files)}: {len(docs)} docs")
    rng.shuffle(docs); docs = docs[:n]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv")
    ap.add_argument("--mode", choices=["extracted", "raw"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_in", type=int, default=2048)
    ap.add_argument("--max_out", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extract_cfg", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--base_model", default="allenai/led-base-16384")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"[train] {out} already trained - skipping"); return

    import torch
    from hyb_gpu import gpu_report, require_gpu
    device = require_gpu()
    from pipeline import clean_text, embed_sentences, extract, free_gpu, set_seed, split_sentences
    from run_systems import DATASETS, load_base_cfg
    set_seed(a.seed); random.seed(a.seed)

    # ---- 1) build (input, target) pairs once and cache them locally
    pairs_p = out / "pairs.jsonl"
    if pairs_p.exists():
        pairs = [json.loads(l) for l in pairs_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        docs = load_train_subset(DATASETS.get(a.dataset, a.dataset), a.split, a.n_train, a.seed, a.cache_dir)
        pairs, cfg = [], load_base_cfg(a.extract_cfg)
        if a.mode == "extracted":
            from sentence_transformers import SentenceTransformer
            sb = SentenceTransformer(a.sbert, device=device)
        for k, d in enumerate(docs):
            sents = split_sentences(d["article"])
            if a.mode == "extracted":
                s = sents[:cfg.max_doc_sentences]
                if not s: continue
                text = extract(s, embed_sentences(s, sb), cfg)["expanded_text"]
            else:
                text = " ".join(sents)
            tgt = clean_text(d["abstract"])
            if text.strip() and tgt.strip():
                pairs.append({"src": text, "tgt": tgt})
            if k % 500 == 0: print(f"  prepared {k}/{len(docs)}")
        if a.mode == "extracted":
            del sb; free_gpu()
        pairs_p.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    print(f"[train] {len(pairs)} pairs, mode={a.mode}, max_in={a.max_in}")

    # ---- 2) fine-tune
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base_model).to(device)
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * math.ceil(len(pairs) / (a.bs * a.accum))
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step, t0, run_loss = 0, time.time(), 0.0
    for ep in range(a.epochs):
        random.shuffle(pairs)
        for b in range(0, len(pairs), a.bs):
            chunk = pairs[b:b + a.bs]
            enc = tok([c["src"] for c in chunk], truncation=True, max_length=a.max_in, padding=True, return_tensors="pt").to(device)
            lab = tok(text_target=[c["tgt"] for c in chunk], truncation=True, max_length=a.max_out, padding=True, return_tensors="pt").input_ids.to(device)
            lab[lab == tok.pad_token_id] = -100
            g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(**enc, global_attention_mask=g, labels=lab).loss / a.accum
            scaler.scale(loss).backward(); run_loss += loss.item()
            if (b // a.bs + 1) % a.accum == 0 or b + a.bs >= len(pairs):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 10 == 0:
                    print(f"  epoch {ep+1} step {step}/{total} loss {run_loss/10:.3f} ({(time.time()-t0)/60:.1f} min)"); run_loss = 0.0
    model.config.use_cache = True
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    (out / "DONE").write_text(json.dumps(vars(a) | {"pairs": len(pairs), "steps": step, "minutes": (time.time() - t0) / 60}, indent=2))
    gpu_report("end of training"); print(f"[train] saved -> {out / 'model'}")


if __name__ == "__main__":
    main()
