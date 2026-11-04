"""
GPU enforcement for every step of the project.

Why this exists: An environment can show "connected to GPU" while the code silently runs on the CPU
(runtime attached to CPU, a pip step replaced the CUDA build of torch, the GPU quota ran out, ...).
The old scripts used `device = "cuda" if torch.cuda.is_available() else "cpu"`, which hides that.
Now every GPU step calls `require_gpu()`, which

  1. fails LOUDLY (never falls back to CPU) if PyTorch cannot see a CUDA device,
  2. runs a real fp16 matrix multiplication on the GPU and reports its TFLOPS,
  3. is followed by `assert_on_device(model)` / `gpu_report()` so the log proves the weights
     live on the GPU and shows peak VRAM actually used.

    python hyb_gpu.py --check                       # standalone check
    python hyb_gpu.py --check --env_out env.json    # + write hardware/software versions (for the paper)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time


def _smi(args):
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def require_gpu(verbose: bool = True) -> str:
    """Return "cuda" or abort the process with an explanation. Never returns "cpu"."""
    import torch
    if not torch.cuda.is_available():
        listed = _smi(["-L"])
        if listed is None:
            why = ("nvidia-smi is not available: this runtime has NO GPU attached.\n"
                   "  -> Runtime > Change runtime type > T4 GPU > Save, then run the cell again.\n"
                   "  -> If no GPU is available, try again later or off-peak.")
        else:
            why = (f"nvidia-smi sees a GPU ({listed.splitlines()[0]}) but PyTorch cannot use it\n"
                   f"  (torch {torch.__version__}, built for CUDA {torch.version.cuda}).\n"
                   "  -> Runtime > Disconnect and delete runtime, reconnect with the T4, and run the cell again.\n"
                   "  -> Do not `pip install torch` yourself in this notebook.")
        raise SystemExit("\n" + "=" * 70 + "\nGPU CHECK FAILED - refusing to run on CPU\n" + why + "\n" + "=" * 70)

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    # a real kernel launch, not just a flag: proves the GPU executes work
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    _ = a @ a
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = a @ a
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096 ** 3 / (time.time() - t0) / 1e12
    del a, _
    torch.cuda.empty_cache()
    if verbose:
        print(f"[gpu] OK: {props.name} | {props.total_memory / 2**30:.1f} GB | torch {torch.__version__} "
              f"(CUDA {torch.version.cuda}) | fp16 matmul {tflops:.1f} TFLOPS")
        if tflops < 3:
            print("[gpu] WARNING: unusually slow GPU (shared or throttled). Results stay valid, runs will be slower.")
    return "cuda"


def assert_on_device(model, device: str, tag: str = "") -> None:
    p = next(model.parameters())
    if p.device.type != device:
        raise RuntimeError(f"[gpu] {tag}: expected weights on {device} but they are on {p.device}")
    print(f"[gpu] {tag}: weights on {p.device}, dtype {str(p.dtype).replace('torch.', '')}")


def gpu_report(tag: str = "") -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB | "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB | "
              f"device in use {(total - free) / 2**30:.2f}/{total / 2**30:.1f} GB")
    except Exception:
        pass


def env_report() -> dict:
    """Hardware + library versions: report these in the paper's experimental-setup section."""
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return None
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "libs": {p: ver(p) for p in ("torch", "transformers", "sentence-transformers", "bert-score",
                                         "rouge-score", "datasets", "numpy", "scipy", "pandas")}}
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": pr.name, "vram_gb": round(pr.total_memory / 2**30, 1),
                           "cuda": torch.version.cuda, "capability": f"{pr.major}.{pr.minor}"}
    except Exception:
        pass
    drv = _smi(["--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        info.setdefault("gpu", {})["driver"] = drv.splitlines()[0]
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--env_out", default=None)
    a = ap.parse_args()
    if a.check:
        require_gpu()
    if a.env_out:
        with open(a.env_out, "w") as f:
            json.dump(env_report(), f, indent=2)
        print(f"[env] hardware/software versions -> {a.env_out}")
    sys.exit(0)
