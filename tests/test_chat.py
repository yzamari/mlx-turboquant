"""Tests for the chat REPL."""

from unittest.mock import MagicMock
import pytest


class TestHandleCommand:
    def _make_repl(self):
        from mlx_turboquant.chat import ChatREPL

        session = MagicMock()
        return ChatREPL(session), session

    def test_quit(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("/quit") == "quit"

    def test_exit(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("/exit") == "quit"

    def test_reset_clears_session(self):
        repl, session = self._make_repl()
        result = repl.handle_command("/reset")
        assert result == "continue"
        session.reset.assert_called_once()

    def test_help(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("/help") == "continue"

    def test_unknown_slash_command(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("/foobar") == "continue"

    def test_regular_text_returns_none(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("hello world") is None

    def test_empty_string_returns_none(self):
        repl, _ = self._make_repl()
        assert repl.handle_command("") is None


class TestFormatStats:
    def test_formats_tokens_and_speed(self):
        from mlx_turboquant.chat import ChatREPL

        line = ChatREPL.format_stats(tokens=50, elapsed=2.5)
        assert "50 tokens" in line
        assert "20.0 tok/s" in line

    def test_zero_elapsed(self):
        from mlx_turboquant.chat import ChatREPL

        line = ChatREPL.format_stats(tokens=10, elapsed=0.0)
        assert "10 tokens" in line


class TestParseArgs:
    def test_default_args(self):
        from mlx_turboquant.chat import build_parser

        args = build_parser().parse_args([])
        assert args.model == "mlx-community/Qwen2.5-3B-Instruct-4bit"
        assert args.key_bits == 3
        assert args.temp == 0.7

    def test_custom_model(self):
        from mlx_turboquant.chat import build_parser

        args = build_parser().parse_args(["--model", "my-model"])
        assert args.model == "my-model"
