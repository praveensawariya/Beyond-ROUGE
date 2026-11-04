#!/usr/bin/env python
"""Phase 2 checkpoint audit (CPU, downloads NO model weights).

Question: is there a released LongT5 / PRIMERA summarization checkpoint
that can be run like-for-like against the study's fine-tuned PEGASUS /
LED-large baselines (same 500-doc test sample, same scoring)?

A pretrained-only checkpoint is NOT a like-for-like baseline and must
not be run as one without saying so. This script only inspects
Hugging Face metadata (model card, config, pipeline tag) and writes
an audit table; the go / no-go decision is made from that table, not
from a generation run.

Usage:
    python checkpoint_audit.py [--out checkpoint_audit.json]

Candidates checked (edit CANDIDATES if a better checkpoint is known,
and record why in the audit note):
    LongT5:  google/long-t5-tglobal-base, google/long-t5-tglobal-large,
             google/long-t5-local-base
    PRIMERA: allenai/PRIMERA, allenai/led-large-16384-arxiv (control:
             this is the study's LED-large, listed only to show the
             audit distinguishes a real checkpoint from a namesake)
"""
import argparse
import json
from pathlib import Path

CANDIDATES = {
    "LongT5": [
        "google/long-t5-tglobal-base",
        "google/long-t5-tglobal-large",
        "google/long-t5-local-base",
    ],
    "PRIMERA": [
        "allenai/PRIMERA",
    ],
}


def audit_one(model_id):
    """Metadata only, via the public Hugging Face API. No weights downloaded."""
    import urllib.error
    import urllib.request

    row = {"model_id": model_id, "exists": False}
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        row["error"] = f"HTTP {e.code} for {url}"
        return row
    except Exception as e:  # noqa: BLE001 - report, don't crash the audit
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["exists"] = True
    row["pipeline_tag"] = info.get("pipeline_tag")
    row["tags"] = [t for t in (info.get("tags") or []) if "summar" in t or "arxiv" in t or "pubmed" in t]
    row["downloads_last_month"] = info.get("downloads")
    cfg = info.get("config") or {}
    row["architectures"] = cfg.get("architectures")
    row["max_position_embeddings"] = cfg.get("max_position_embeddings")
    row["vocab_size"] = cfg.get("vocab_size")
    card = info.get("cardData") or {}
    row["card_datasets"] = card.get("datasets")
    row["card_language"] = card.get("language")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoint_audit.json")
    a = ap.parse_args()
    results = {}
    for family, ids in CANDIDATES.items():
        results[family] = [audit_one(mid) for mid in ids]
    Path(a.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    for family, rows in results.items():
        print(f"\n== {family} ==")
        for r in rows:
            if not r["exists"]:
                print(f"  {r['model_id']}: NOT FOUND / error: {r.get('error', '')[:160]}")
            else:
                print(f"  {r['model_id']}: pipeline={r['pipeline_tag']} arch={r['architectures']} "
                      f"max_pos={r['max_position_embeddings']} tags={r['tags']}")
    print(f"\nwrote {a.out}")
    print("DECISION RULE (apply by hand, do not automate away):")
    print("  GO only if a checkpoint is a summarization fine-tune with a")
    print("  documented arXiv/PubMed (or long-document) setup. Pretrained-only")
    print("  => documented exclusion in the rebuttal, NOT a zero-shot run")
    print("  labelled as the baseline.")


if __name__ == "__main__":
    main()
