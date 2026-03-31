"""
Long-context stress test: how far can TurboQuant push your Mac?

Tests progressively longer prompts (4K → 128K tokens) with and without
TurboQuant, reporting speed and peak memory at each step. Stops when
the system runs out of memory.

Usage:
    python benchmarks/bench_long_context.py
    python benchmarks/bench_long_context.py --model mlx-community/Qwen2.5-32B-Instruct-4bit
    python benchmarks/bench_long_context.py --contexts 4096 8192 16384 32768
"""

import argparse
import time
import sys

import mlx.core as mx


FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending "
    "reports, and projected revenue streams from various business units. The "
    "committee discussed infrastructure upgrades planned for the western "
    "regional offices and noted that maintenance schedules should be coordinated "
    "with the facilities management team. Several action items were assigned to "
    "team leads for follow-up before the next meeting cycle. "
)

QUESTION = "\n\nBased on the above, write a brief summary."


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens length."""
    filler_ids = tokenizer.encode(FILLER)
    filler_len = len(filler_ids)
    n_reps = max(1, target_tokens // filler_len)

    text = FILLER * n_reps + QUESTION
    # Trim to approximate target
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        text = tokenizer.decode(ids[:target_tokens])

    return text


def run_single(model, tokenizer, prompt_text, max_tokens, prompt_cache=None):
    """Run generation, return metrics dict or None on failure."""
    import mlx_lm

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt_text
    else:
        formatted = prompt_text

    gen_kwargs = dict(max_tokens=max_tokens)
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache

    mx.reset_peak_memory()
    try:
        start = time.perf_counter()
        last = None
        for response in mlx_lm.stream_generate(
            model, tokenizer, formatted, **gen_kwargs
        ):
            last = response
        elapsed = time.perf_counter() - start
    except Exception as e:
        return None, str(e)

    peak_mb = mx.get_peak_memory() / 1e6
    cache_mb = 0
    if prompt_cache is not None:
        cache_mb = sum(c.nbytes for c in prompt_cache) / 1e6

    return {
        "prompt_tokens": last.prompt_tokens,
        "gen_tokens": last.generation_tokens,
        "prompt_tps": last.prompt_tps,
        "gen_tps": last.generation_tps,
        "peak_mb": peak_mb,
        "cache_mb": cache_mb,
        "elapsed": elapsed,
    }, None


def main():
    parser = argparse.ArgumentParser(
        description="Long-context stress test: TurboQuant vs Standard",
    )
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="Context lengths to test (default: 4K 8K 16K 32K 64K 128K)",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=128)
    args = parser.parse_args()

    import mlx_lm
    from mlx_turboquant.patch import make_turboquant_cache

    print(f"Model: {args.model}")
    print(f"Contexts: {args.contexts}")
    print(f"Max generation tokens: {args.max_tokens}")
    print(f"TurboQuant: key_bits={args.key_bits}, value_bits={args.value_bits}, buffer={args.buffer_size}")
    print()

    print("Loading model...")
    model, tokenizer = mlx_lm.load(args.model)
    print("Ready!\n")

    # Header
    print(f"{'Context':>8} │ {'Mode':<12} │ {'Prompt tok/s':>12} │ {'Gen tok/s':>10} │ {'Peak MB':>8} │ {'Cache MB':>9}")
    print("─" * 75)

    results = []

    for ctx in args.contexts:
        prompt = build_prompt(tokenizer, ctx)
        actual_tokens = len(tokenizer.encode(prompt))

        # --- Standard ---
        r_std, err = run_single(model, tokenizer, prompt, args.max_tokens)
        if r_std is None:
            print(f"{actual_tokens:>8} │ {'Standard':<12} │ {'OOM: ' + err[:30]:>50}")
            # If standard fails, TurboQuant might still work
        else:
            print(
                f"{actual_tokens:>8} │ {'Standard':<12} │ "
                f"{r_std['prompt_tps']:>12.1f} │ {r_std['gen_tps']:>10.1f} │ "
                f"{r_std['peak_mb']:>8.0f} │ {'—':>9}"
            )

        # --- TurboQuant ---
        tq_cache = make_turboquant_cache(
            model,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            buffer_size=args.buffer_size,
        )
        r_tq, err = run_single(
            model, tokenizer, prompt, args.max_tokens, prompt_cache=tq_cache
        )
        if r_tq is None:
            print(f"{actual_tokens:>8} │ {'TurboQuant':<12} │ {'OOM: ' + err[:30]:>50}")
            print()
            print("Stopping — out of memory.")
            break
        else:
            speedup = ""
            if r_std and r_std["gen_tps"] > 0:
                ratio = r_tq["gen_tps"] / r_std["gen_tps"]
                speedup = f" ({ratio:.1f}x)"
            print(
                f"{'':>8} │ {'TurboQuant':<12} │ "
                f"{r_tq['prompt_tps']:>12.1f} │ {r_tq['gen_tps']:>10.1f} │ "
                f"{r_tq['peak_mb']:>8.0f} │ {r_tq['cache_mb']:>8.0f}{'':1}"
                f"{speedup}"
            )

        results.append({
            "context": actual_tokens,
            "standard": r_std,
            "turboquant": r_tq,
        })
        print("─" * 75)

    # Summary
    if results:
        print()
        print("SUMMARY")
        print("─" * 60)
        for r in results:
            ctx = r["context"]
            std = r["standard"]
            tq = r["turboquant"]
            if std and tq:
                speed_ratio = tq["gen_tps"] / std["gen_tps"] if std["gen_tps"] > 0 else 0
                mem_saved = std["peak_mb"] - tq["peak_mb"]
                print(
                    f"  {ctx:>6} tokens: TurboQuant is {speed_ratio:.1f}x gen speed, "
                    f"{mem_saved:+.0f} MB peak memory"
                )
            elif tq and not std:
                print(f"  {ctx:>6} tokens: Standard OOM, TurboQuant succeeded ({tq['peak_mb']:.0f} MB)")


if __name__ == "__main__":
    main()
