"""
Dataset access tuned for small downloads.

* Downloads ONLY the parquet shard(s) of the split you ask for (arXiv test ~0.1 GB) instead of
  letting `load_dataset` fetch and rebuild every split (train alone is several GB).
* Caches the selected subset as a small JSONL (point HYB_DATA_CACHE / --cache_dir at a local directory),
  so a fresh session reads it in a second instead of re-downloading.
* Selection is identical to the original `rng.permutation(len(ds))[:n]`, so bigger --n runs are
  nested supersets and reuse every prediction already computed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# repos that ship plain parquet shards named  <config>/<split>-XXXXX-of-XXXXX.parquet
PARQUET_CONFIG = {"ccdv/arxiv-summarization": "section", "ccdv/pubmed-summarization": "section"}


def _table_from_parquet(repo, split, config, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"{config}/{split}-") and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards for {repo}:{config}/{split}")
    print(f"downloading {len(files)} parquet shard(s) of {repo} [{split}] ...")
    parts = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=columns)
             for f in files]
    return pa.concat_tables(parts)


def _select(repo, split, text_col, summary_col, n, seed):
    src = None
    if repo in PARQUET_CONFIG:
        try:
            tbl = _table_from_parquet(repo, split, PARQUET_CONFIG[repo], [text_col, summary_col])
            src = ("arrow", tbl, tbl.num_rows)
        except Exception as e:                                   # network/layout change -> generic path
            print(f"parquet shortcut failed ({type(e).__name__}: {e}); using datasets.load_dataset")
    if src is None:
        from datasets import load_dataset
        ds = load_dataset(repo, split=split)
        src = ("hf", ds, len(ds))
    kind, obj, total = src
    idxs = [int(i) for i in np.random.default_rng(seed).permutation(total)[:n]]
    if kind == "arrow":
        t = obj.take(idxs)
        arts, abss = t[text_col].to_pylist(), t[summary_col].to_pylist()
    else:
        sub = obj.select(idxs)
        arts, abss = sub[text_col], sub[summary_col]
    return [{"idx": i, "article": a, "abstract": b} for i, a, b in zip(idxs, arts, abss)]


def _read_cache(path: Path):
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:                         # partial write after a disconnect
                return []
    return docs


def load_subset(repo, split, n, seed, text_col="article", summary_col="abstract", cache_dir=None):
    """-> list of {"idx", "article", "abstract"} in permutation order (first n of the seeded shuffle)."""
    cache_dir = cache_dir or os.environ.get("HYB_DATA_CACHE")
    path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        name = f"{repo.replace('/', '__')}__{split}__seed{seed}__{text_col}__{summary_col}.jsonl"
        path = Path(cache_dir) / name
        if path.exists():
            docs = _read_cache(path)
            if len(docs) >= n:
                print(f"[data] {len(docs[:n])} {split} docs from cache {path.name}")
                return docs[:n]
    docs = _select(repo, split, text_col, summary_col, n, seed)
    if path is not None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        os.replace(tmp, path)
        print(f"[data] cached {len(docs)} {split} docs -> {path}")
    return docs
