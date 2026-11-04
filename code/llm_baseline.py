#!/usr/bin/env python
"""Zero-shot open-LLM baseline (writes preds in the same format as run_systems.py, so analyze.py can score it).
Needs a GPU with >= 24 GB (A100/L4/RTX 4090). Greedy decoding, same 300-token cap as every other system."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--name", default="llm_qwen7b")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_in_tokens", type=int, default=6000)
    a = ap.parse_args()
    import torch
    from hyb_gpu import require_gpu
    device = require_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    res = Path(a.results)
    refs = [json.loads(l) for l in (res / "refs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:a.n]
    outp = res / "preds" / f"{a.name}.jsonl"
    done = {json.loads(l)["idx"] for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()} if outp.exists() else set()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    prompt = ("Write the abstract (150-200 words) of the following scientific paper. Use only information that is "
              "stated in the paper; do not add facts or numbers. Output only the abstract.\n\nPAPER:\n{paper}")
    with open(outp, "a", encoding="utf-8") as f:
        for k, r in enumerate(refs):
            if r["idx"] in done: continue
            ids = tok(r["article"], add_special_tokens=False).input_ids[:a.max_in_tokens]
            msg = [{"role": "user", "content": prompt.format(paper=tok.decode(ids))}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(device); t0 = time.time()
            with torch.inference_mode():
                o = model.generate(**enc, max_new_tokens=300, do_sample=False)
            pred = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()  # dataset text is lower-cased
            f.write(json.dumps({"idx": r["idx"], "pred": pred, "sec": round(time.time() - t0, 3)}) + "\n"); f.flush()
            if k % 10 == 0: print(f"  {k+1}/{len(refs)}")


if __name__ == "__main__":
    main()
