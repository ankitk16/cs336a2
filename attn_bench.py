"""Benchmark plain (single-head) attention across d_model x sequence length.

    python -m cs336_systems.attention_bench
    python -m cs336_systems.attention_bench --dtype bf16
    python -m cs336_systems.attention_bench --no-causal --steps 50

Batch is fixed at 8 and there is no head dimension, so Q, K, V are
(batch, seq, d_model) and the score matrix is (batch, seq, seq).

Two things worth knowing about how the timing is done:

* The 100 backward passes reuse one graph via retain_graph=True. Calling
  backward() normally frees the graph, so a second call would raise. Reusing it
  keeps the saved activations alive -- which is exactly the state we want to
  measure anyway -- and isolates the backward kernels from the forward. The
  gradients accumulate across the 100 calls; the values are meaningless but the
  timing is not.

* Memory is read after a synchronize, between the forward and the first
  backward, which is the peak of the forward pass: scores, softmax output and
  the mask are all live and nothing has been freed yet.
"""

from __future__ import annotations

import math

import torch

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BATCH = 8

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """Scaled dot-product attention, no heads. The textbook three lines."""
    d = q.shape[-1]
    s = q @ k.transpose(-2, -1) / math.sqrt(d)
    # PyTorch swaps the last two dimensions of tensor k.

    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    return p @ v


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_one(d_model: int, seq: int, device: torch.device, args):
    dtype = DTYPES[args.dtype]

    gen = torch.Generator(device=device).manual_seed(args.seed)

    def rand():
        return torch.randn(
            BATCH,
            seq,
            d_model,
            device=device,
            dtype=dtype,
            generator=gen,
            requires_grad=True,
        )

    q, k, v = rand(), rand(), rand()

    mask = (
        torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), 1)
        if args.causal
        else None
    )

    # ----------------Forward---------------#
    for _ in range(args.warmup):
        attention(q, k, v, mask)
    sync(device)

    import time

    t0 = time.perf_counter()

    for _ in range(args.steps):
        o = attention(q, k, v, mask)
    sync(device)

    fwd_ms = (time.perf_counter() - t0) / args.steps * 1e3

    # ---- memory in use with the forward graph alive -----------------------#
    loss = o.sum()
    sync(device)

    mem_gb_fwd = (
        torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )

    # --------------backward-------------------#
    for _ in range(args.warmup):
        loss.backward(retain_graph=True)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        o = attention(q, k, v, mask)
        loss = o.sum()
        loss.backward()
    sync(device)
    bwd_ms = (time.perf_counter() - t0) / args.steps * 1e3
    mem_gb_bwd = (
        torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )

    return fwd_ms, mem_gb_fwd, mem_gb_bwd, bwd_ms


def predicted_gb(d_model, seq, width, causal):
    """Analytic memory for the live tensors at the forward/backward boundary. Width is bytes for unit.
    That would be 4 bytes for fp32 and so on."""

    qkvo = 4 * BATCH * seq * d_model * width  # q, k,v, output
    scores = BATCH * seq * seq * width  # 2 since pre and post softmax
    maskb = seq * seq * 1 if causal else 0
    return (qkvo + scores + maskb) / 2**30


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-models", type=int, nargs="+", default=D_MODELS)
    p.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    args = p.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    width = torch.finfo(DTYPES[args.dtype]).bits // 8

    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    total = (
        torch.cuda.get_device_properties(device).total_memory / 2**30
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"{name}  {total:.0f} GiB  |  torch {torch.__version__}  |  "
        f"batch {BATCH}  dtype {args.dtype}  causal {args.causal}  steps {args.steps}\n"
    )

    print(
        f"{'d_model':>8}{'seq':>7}{'fwd ms':>10}{'bwd ms':>10}"
        f"{'mem GB fwd':>12}{'predicted':>11}{'mem GB bwd':>12}"
    )
    import itertools

    for d_model, seq in itertools.product(args.d_models, args.seq_lens):
        pred = predicted_gb(d_model, seq, width, args.causal)

        try:
            fwd, mem_fwd, mem_bwd, bwd = bench_one(d_model, seq, device, args)
        except torch.OutOfMemoryError:
            print(
                f"{d_model:>8}{seq:>7}{'OOM':>10}{'':>10}{'':>12}{pred:>11.2f}{'':>12}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        print(
            f"{d_model:>8}{seq:>7}{fwd:>10.2f}{bwd:>10.2f}{mem_fwd:>12.2f}{pred:>11.2f} {mem_bwd:>12.2f}"
        )

        del fwd, bwd, mem_fwd, mem_bwd
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
