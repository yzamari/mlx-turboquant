"""
Interactive chat REPL with TurboQuant KV cache compression.

Usage:
    mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit
    mlx-tq-chat --model mlx-community/Qwen2.5-32B-Instruct-4bit --key-bits 3
"""

import argparse
import time

from mlx_turboquant._session import InferenceSession


COMMANDS = {
    "/quit": "Exit the chat",
    "/exit": "Exit the chat",
    "/reset": "Clear conversation history",
    "/help": "Show available commands",
}


class ChatREPL:
    """Terminal chat loop with slash commands and streaming output."""

    def __init__(self, session: InferenceSession):
        self.session = session

    def handle_command(self, text: str) -> str | None:
        """Process slash commands.

        Returns:
            "quit" — caller should exit
            "continue" — command handled, prompt again
            None — not a command, treat as user message
        """
        stripped = text.strip()
        if not stripped:
            return None

        lower = stripped.lower()
        if lower in ("/quit", "/exit"):
            return "quit"
        if lower == "/reset":
            self.session.reset()
            print("\nConversation cleared.\n")
            return "continue"
        if lower == "/help":
            print("\nCommands:")
            for cmd, desc in COMMANDS.items():
                print(f"  {cmd:12s} {desc}")
            print()
            return "continue"
        if stripped.startswith("/"):
            print(f"\nUnknown command: {stripped}. Type /help for commands.\n")
            return "continue"
        return None

    @staticmethod
    def format_stats(tokens: int, elapsed: float) -> str:
        tps = tokens / elapsed if elapsed > 0 else 0
        return f"[{tokens} tokens, {tps:.1f} tok/s, {elapsed:.1f}s]"

    def run(self, max_tokens: int = 1024, temp: float = 0.7, top_p: float = 0.9):
        """Main chat loop. Blocks until user quits."""
        model_name = self.session.model_path.split("/")[-1]
        tq_status = "ON" if self.session.use_turboquant else "OFF"
        print(f"\nChat with {model_name} (TurboQuant {tq_status})")
        print("Type /help for commands, /quit to exit.\n")

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue

            action = self.handle_command(user_input)
            if action == "quit":
                print("Bye!")
                break
            if action == "continue":
                continue

            start = time.perf_counter()
            gen_tokens = 0
            try:
                for response in self.session.generate_response(
                    user_input, max_tokens=max_tokens, temp=temp, top_p=top_p,
                ):
                    print(response.text, end="", flush=True)
                    gen_tokens = response.generation_tokens
            except KeyboardInterrupt:
                pass  # Ctrl+C stops generation gracefully

            elapsed = time.perf_counter() - start
            print(f"\n{self.format_stats(gen_tokens, elapsed)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with an LLM using TurboQuant KV cache compression",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="HuggingFace model path or local directory (default: Qwen2.5-3B-4bit)",
    )
    parser.add_argument(
        "--key-bits", type=int, default=3, choices=[2, 3, 4],
        help="Key compression bits (default: 3)",
    )
    parser.add_argument(
        "--value-bits", type=int, default=2, choices=[2, 4],
        help="Value compression bits (default: 2)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=128,
        help="Recent tokens kept uncompressed (default: 128)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--no-turboquant", action="store_true",
        help="Disable TurboQuant (standard KV cache)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    session = InferenceSession(
        model_path=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        use_turboquant=not args.no_turboquant,
    )

    print(f"Loading {args.model}...")
    session.load()
    print("Ready!")

    repl = ChatREPL(session)
    repl.run(max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)


if __name__ == "__main__":
    main()
