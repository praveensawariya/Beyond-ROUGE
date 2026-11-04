#!/usr/bin/env python
"""End-to-end latency and peak GPU memory benchmark per system.

Latency includes sentence splitting, SBERT embedding and extractive selection
where applicable, plus generation where the system uses a generative model.
Model loading is excluded from per-document latency. One document is processed
as a warm-up and excluded from latency statistics.

The output CSV records both the requested and actual number of timed documents.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def load_reference_records(results_dir: Path, requested_docs: int):
    """Load one warm-up record plus the requested number of timed records."""
    refs_path = results_dir / "refs.jsonl"
    if not refs_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {refs_path}")

    refs = []
    with refs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {refs_path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected each JSONL record to be an object; line {line_number} is not."
                )
            if "article" not in record or not isinstance(record["article"], str):
                raise ValueError(
                    f"Record at line {line_number} must contain an 'article' string."
                )
            refs.append(record)

    required_records = requested_docs + 1  # one warm-up + requested timed documents
    if len(refs) < required_records:
        raise ValueError(
            f"Insufficient records in {refs_path}: requested {requested_docs} timed "
            f"documents plus 1 warm-up ({required_records} records total), but only "
            f"{len(refs)} valid records were found. Reduce --n or provide more records."
        )

    return refs[:required_records]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end per-document latency and peak GPU memory."
    )
    parser.add_argument("--results", required=True, help="Directory containing refs.jsonl")
    parser.add_argument("--systems", nargs="+", required=True, help="System names to benchmark")
    parser.add_argument("--extract_cfg", required=True, help="Path to extraction configuration")
    parser.add_argument("--n", type=int, default=20, help="Number of timed documents (default: 20)")
    parser.add_argument(
        "--led_tokens", type=int, default=4096,
        help="Input token limit for systems configured to use led_tokens (default: 4096)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1 so there is at least one timed document.")
    if args.led_tokens < 1:
        parser.error("--led_tokens must be a positive integer.")

    results_dir = Path(args.results)
    refs = load_reference_records(results_dir, args.n)

    # Make sibling project modules importable regardless of the current directory.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import pandas as pd
    import torch
    from hyb_gpu import require_gpu
    from pipeline import Seq2Seq, embed_sentences, extract, free_gpu, make_gen_kwargs, split_sentences
    from run_systems import MODELS, SYSTEMS, load_base_cfg, make_variants
    from sentence_transformers import SentenceTransformer

    unknown_systems = [name for name in args.systems if name not in SYSTEMS]
    if unknown_systems:
        raise ValueError(
            f"Unknown system name(s): {unknown_systems}. Available systems: "
            f"{sorted(SYSTEMS.keys())}"
        )

    device = require_gpu()
    variants = make_variants(load_base_cfg(args.extract_cfg))
    generation_kwargs = make_gen_kwargs(4)

    print(f"Reference file: {results_dir / 'refs.jsonl'}")
    print(f"Requested timed documents per system: {args.n}")
    print("Warm-up documents per system: 1 (excluded from latency statistics)")
    print(f"Records loaded and validated: {len(refs)}")
    print(f"Device: {device}")

    # The embedding model is initialized once and reused across systems.
    sentence_bert = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    rows = []

    for system_name in args.systems:
        print(f"\nBenchmarking system: {system_name}")
        model_key, variant_name, output_field, max_input = SYSTEMS[system_name]
        if max_input == "led_tokens":
            max_input = args.led_tokens

        model = None
        try:
            if model_key:
                model = Seq2Seq(*MODELS[model_key], device)
                parameter_count_millions = sum(
                    parameter.numel() for parameter in model.model.parameters()
                ) / 1e6
            else:
                parameter_count_millions = 0.0

            # Reset immediately before this system's measured run. Peak memory
            # includes the warm-up document and all timed documents for this system.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            elapsed_seconds = []

            for record_index, record in enumerate(refs):
                start_time = time.perf_counter()
                sentences = split_sentences(record["article"])

                if variant_name in (None, "raw"):
                    text_to_summarize = " ".join(sentences)
                else:
                    if variant_name not in variants:
                        raise KeyError(
                            f"Extraction variant {variant_name!r} for system "
                            f"{system_name!r} was not found in the loaded configuration."
                        )
                    config = variants[variant_name]
                    selected_sentences = sentences[: config.max_doc_sentences]
                    embeddings = embed_sentences(selected_sentences, sentence_bert)
                    extracted = extract(selected_sentences, embeddings, config)
                    if output_field not in extracted:
                        raise KeyError(
                            f"Output field {output_field!r} was not returned by extract() "
                            f"for system {system_name!r}. Available fields: "
                            f"{list(extracted.keys())}"
                        )
                    text_to_summarize = extracted[output_field]

                if system_name == "lead":
                    text_to_summarize = " ".join(text_to_summarize.split()[:200])

                if model is not None:
                    model.summarize(text_to_summarize, max_input, generation_kwargs)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                # First document is warm-up; all remaining records are timed.
                if record_index > 0:
                    elapsed_seconds.append(elapsed)

            actual_timed_docs = len(elapsed_seconds)
            if actual_timed_docs != args.n:
                raise RuntimeError(
                    f"{system_name}: expected {args.n} timed documents after warm-up, "
                    f"but measured {actual_timed_docs}. No result row was written for this system."
                )

            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            row = {
                "system": system_name,
                "params_M": round(parameter_count_millions, 3),
                "requested_docs": args.n,
                "actual_timed_docs": actual_timed_docs,
                "warmup_docs": 1,
                "sec_per_doc_mean": sum(elapsed_seconds) / actual_timed_docs,
                "sec_per_doc_median": sorted(elapsed_seconds)[actual_timed_docs // 2],
                "peak_vram_GB": peak_memory_gb,
            }
            rows.append(row)
            print(row)

        finally:
            if model is not None:
                model.close()
            free_gpu()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(rows)
    output_df.round(3).to_csv(output_path, index=False)

    print("\nBenchmark complete.")
    print(f"Systems successfully recorded: {len(rows)} / {len(args.systems)}")
    print(f"Timed documents per successful system: {args.n}")
    print(f"Output CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
