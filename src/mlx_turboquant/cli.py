"""
CLI entry point for mlx-tq-generate.

Usage:
    mlx-tq-generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
                     --prompt "Explain quantum computing" \\
                     --key-bits 3 --max-tokens 200
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with TurboQuant KV cache compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlx-tq-generate --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Hello world"
  mlx-tq-generate --model Qwen/Qwen3-8B --prompt "Explain AI" --key-bits 3 --max-tokens 200
  mlx-tq-generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "Write a poem" --no-turboquant
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="HuggingFace model path or local model directory",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Key compression bits (default: 3)",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="Value compression bits (default: 2)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=128,
        help="Number of recent tokens kept uncompressed (default: 128)",
    )
    parser.add_argument(
        "--value-group-size",
        type=int,
        default=32,
        help="Group size for value quantization (default: 32)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--no-turboquant",
        action="store_true",
        help="Disable TurboQuant (use standard KV cache for comparison)",
    )

    args = parser.parse_args()

    from mlx_turboquant.generate import generate

    result = generate(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        value_group_size=args.value_group_size,
        temp=args.temp,
        verbose=True,
        use_turboquant=not args.no_turboquant,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
