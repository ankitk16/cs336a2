from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from cs336_basics.nn_modules import Transformer, cross_entropy

CONFIGS: dict[str, dict[str, int]] = {
    "toy": {"d_model": 256, "d_ff": 1024, "num_layers": 4, "num_heads": 8},
    # the handout's sizes
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, warmup: int, steps: int, device: torch.device) -> float:
    """
    Mean seconds per call. Warmup discarded; sync outside the timed loop.
    """
    for _ in range(warmup):
        fn()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    sync(device)

    return (time.perf_counter() - t0) / steps


def make_steps(model, x, y, dtype: str, device: torch.device):
    """Three closures: forward, forward+loss, forward+loss+backward."""
    amp = (
        torch.autocast(device.type, dtype=DTYPES[dtype])
        if dtype != "fp32" and device.type == "cuda"
        else None
    )

    def forward():
        if amp is None:
            model(x)
        else:
            with amp:
                model(x)

    def fwd_loss():
        if amp is None:
            cross_entropy(model(x), y)
        else:
            with amp:
                logits = model(x)
            cross_entropy(logits.float(), y)

    def full():
        model.zero_grad(set_to_none=True)
        if amp is None:
            logits = model(x)
            loss = cross_entropy(logits, y)

        else:
            with amp:
                logits = model(x)
            loss = cross_entropy(logits.float(), y)

        loss.backward()

    return {"fwd": forward, "fwd+loss": fwd_loss, "full": full}


def run_one(size, compiled, dtype, args, device) -> dict | None:
    cfg = CONFIGS[size]
    torch.manual_seed(args.seed)

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        theta=args.rope_theta,
        **cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    if compiled:
        model = torch.compile(model)

    x = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )
    y = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=device
    )

    steps = make_steps(model, x, y, dtype, device)

    warmup = args.warmup + (10 if compiled else 0)
    out = {"size": size, "compile": compiled, "dtype": dtype, "n_params": n_params}

    for name, fn in steps.items():
        out[name] = timeit(fn, warmup, args.steps, device) * 1e3

    out["peak_gb"] = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )

    return out


HEADER = (
    f"{'size':<8}{'compile':<9}{'dtype':<7}{'params':>9}"
    f"{'fwd ms':>10}{'+loss':>9}{'full':>9}{'peak GB':>9}"
)


def row(r: dict) -> str:
    return (
        f"{r['size']:<8}{'yes' if r['compile'] else 'no':<9}{r['dtype']:<7}"
        f"{r['n_params'] / 1e6:>8.0f}M"
        f"{r['fwd']:>10.1f}{r['fwd+loss']:>9.1f}{r['full']:>9.1f}{r['peak_gb']:>9.2f}"
    )


import argparse


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["toy"], choices=list(CONFIGS))
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--no-compile", action="store_true", help="skip the compiled rows")
    p.add_argument(
        "--dtypes", nargs="+", default=["fp32", "bf16"], choices=list(DTYPES)
    )
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = get_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    name = torch.cuda.get_device_name if device.type == "cuda" else "cpu"

    print(
        f"{name}  torch {torch.__version__}"
        f"batch {args.batch_size} x ctx {args.context_length}\n"
    )

    compiles = [False] if args.no_compile else [False, True]
    print(HEADER)
    results = []
    for size in args.sizes:
        for compiled in compiles:
            for dtype in args.dtypes:
                try:
                    r = run_one(size, compiled, dtype, args, device)
                except torch.OutOfMemoryError:
                    print(
                        f"{size:<8}{'yes' if compiled else 'no':<9}{dtype:<7}"
                        f"{'OOM -- lower --batch-size':>37}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                results.append(r)
                print(row(r), flush=True)

    if args.csv and results:
        path = Path(args.csv)
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            if new:
                w.writeheader()
            w.writerows(results)
        print(f"\nappended {len(results)} rows to {path}")
    return 0


# running the whole file
if __name__ == "__main__":
    raise SystemExit(main())
