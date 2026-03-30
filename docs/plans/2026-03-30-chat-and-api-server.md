# Chat REPL + OpenAI-Compatible API Server

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an Ollama-like interactive chat and an OpenAI-compatible HTTP API to mlx-turboquant, so users can chat with any MLX model using TurboQuant KV cache compression.

**Architecture:** Two new modules — `chat.py` (terminal REPL with multi-turn conversation) and `server.py` (FastAPI server implementing `/v1/chat/completions`). Both reuse the existing `generate.py` and `patch.py` infrastructure. A shared `_session.py` module holds the loaded model + cache state to avoid duplication.

**Tech Stack:** mlx-lm (model loading/generation), FastAPI + uvicorn (API server), existing TurboQuant Metal kernels

---

## Task 1: Shared Session Module

The session holds a loaded model, tokenizer, and TurboQuant config so both chat and server can reuse it without duplicating model-loading logic.

**Files:**
- Create: `src/mlx_turboquant/_session.py`
- Test: `tests/test_session.py`

**Step 1: Write the failing test**

```python
# tests/test_session.py
"""Tests for the shared inference session."""

from unittest.mock import patch, MagicMock
import pytest


def test_session_init_stores_config():
    """Session stores TurboQuant config without loading model yet."""
    from mlx_turboquant._session import InferenceSession

    session = InferenceSession(
        model_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
        key_bits=3,
        value_bits=2,
        buffer_size=128,
    )
    assert session.model_path == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert session.key_bits == 3
    assert session.model is None  # not loaded yet


def test_session_load_calls_mlx_lm(monkeypatch):
    """Session.load() calls mlx_lm.load and creates TurboQuant cache."""
    from mlx_turboquant._session import InferenceSession

    fake_model = MagicMock()
    fake_model.layers = [MagicMock() for _ in range(4)]
    # Give each layer an attention module with head_dim
    for layer in fake_model.layers:
        layer.self_attn = MagicMock()
        layer.self_attn.head_dim = 64
        layer.is_linear = False

    fake_tokenizer = MagicMock()
    monkeypatch.setattr("mlx_lm.load", lambda path: (fake_model, fake_tokenizer))

    session = InferenceSession(
        model_path="test-model", key_bits=3, value_bits=2, buffer_size=64
    )
    session.load()

    assert session.model is fake_model
    assert session.tokenizer is fake_tokenizer


def test_session_reset_cache():
    """reset_cache() clears conversation state for new chat."""
    from mlx_turboquant._session import InferenceSession

    session = InferenceSession(model_path="test-model")
    session._messages = [{"role": "user", "content": "hello"}]
    session.reset_cache()
    assert session._messages == []
    assert session._prompt_cache is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_turboquant._session'`

**Step 3: Write minimal implementation**

```python
# src/mlx_turboquant/_session.py
"""
Shared inference session: loads model once, manages conversation state.

Used by both the chat REPL and the API server.
"""

from typing import Optional

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler


class InferenceSession:
    """Holds a loaded model, tokenizer, and TurboQuant cache config."""

    def __init__(
        self,
        model_path: str,
        key_bits: int = 3,
        value_bits: int = 2,
        buffer_size: int = 128,
        value_group_size: int = 32,
        temp: float = 0.7,
        use_turboquant: bool = True,
    ):
        self.model_path = model_path
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.value_group_size = value_group_size
        self.temp = temp
        self.use_turboquant = use_turboquant

        self.model = None
        self.tokenizer = None
        self._prompt_cache = None
        self._messages: list[dict] = []

    def load(self):
        """Load model and tokenizer. Call once before generating."""
        import mlx_lm
        from mlx_turboquant.patch import make_turboquant_cache

        self.model, self.tokenizer = mlx_lm.load(self.model_path)

        if self.use_turboquant:
            self._prompt_cache = make_turboquant_cache(
                self.model,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                value_group_size=self.value_group_size,
                buffer_size=self.buffer_size,
            )

    def reset_cache(self):
        """Clear conversation history and KV cache for a new chat."""
        self._messages = []
        self._prompt_cache = None

    def generate_response(self, user_message: str, max_tokens: int = 1024):
        """
        Add user message, generate assistant response, yield token strings.

        Yields text chunks as they are generated (streaming).
        """
        import mlx_lm

        self._messages.append({"role": "user", "content": user_message})

        formatted = self.tokenizer.apply_chat_template(
            self._messages, tokenize=False, add_generation_prompt=True
        )

        # Build fresh cache for full conversation each turn
        # (multi-turn with incremental cache is a future optimization)
        if self.use_turboquant:
            from mlx_turboquant.patch import make_turboquant_cache
            self._prompt_cache = make_turboquant_cache(
                self.model,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                value_group_size=self.value_group_size,
                buffer_size=self.buffer_size,
            )

        sampler = make_sampler(temp=self.temp)
        gen_kwargs = dict(max_tokens=max_tokens, sampler=sampler)
        if self._prompt_cache is not None:
            gen_kwargs["prompt_cache"] = self._prompt_cache

        full_text = ""
        for response in mlx_lm.stream_generate(
            self.model, self.tokenizer, formatted, **gen_kwargs
        ):
            yield response.text
            full_text += response.text

        self._messages.append({"role": "assistant", "content": full_text})
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/test_session.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/mlx_turboquant/_session.py tests/test_session.py
git commit -m "feat: add shared InferenceSession for chat and API"
```

---

## Task 2: Chat REPL

Interactive terminal chat — type messages, get streaming responses, `/reset` to clear history, `/quit` to exit. Like `ollama run`.

**Files:**
- Create: `src/mlx_turboquant/chat.py`
- Test: `tests/test_chat.py`
- Modify: `pyproject.toml` (add `mlx-tq-chat` entry point)

**Step 1: Write the failing test**

```python
# tests/test_chat.py
"""Tests for the chat REPL module."""

from unittest.mock import MagicMock, patch
from io import StringIO
import pytest


def test_handle_slash_reset():
    """'/reset' clears conversation and prints confirmation."""
    from mlx_turboquant.chat import ChatREPL
    from mlx_turboquant._session import InferenceSession

    session = MagicMock(spec=InferenceSession)
    repl = ChatREPL(session)

    result = repl.handle_command("/reset")
    assert result == "continue"
    session.reset_cache.assert_called_once()


def test_handle_slash_quit():
    """'/quit' signals exit."""
    from mlx_turboquant.chat import ChatREPL
    from mlx_turboquant._session import InferenceSession

    session = MagicMock(spec=InferenceSession)
    repl = ChatREPL(session)

    result = repl.handle_command("/quit")
    assert result == "quit"


def test_handle_regular_message():
    """Regular text is not a command."""
    from mlx_turboquant.chat import ChatREPL
    from mlx_turboquant._session import InferenceSession

    session = MagicMock(spec=InferenceSession)
    repl = ChatREPL(session)

    result = repl.handle_command("hello world")
    assert result is None  # not a command


def test_format_stats():
    """Stats line formats token count and speed."""
    from mlx_turboquant.chat import ChatREPL

    line = ChatREPL.format_stats(tokens=50, elapsed=2.5)
    assert "50 tokens" in line
    assert "20.0 tok/s" in line
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/test_chat.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/mlx_turboquant/chat.py
"""
Interactive chat REPL with TurboQuant KV cache compression.

Usage:
    mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit
    mlx-tq-chat --model mlx-community/Qwen2.5-32B-Instruct-4bit --key-bits 3
"""

import sys
import time
import argparse

from mlx_turboquant._session import InferenceSession


class ChatREPL:
    """Terminal chat loop with slash commands."""

    COMMANDS = {
        "/quit": "Exit the chat",
        "/exit": "Exit the chat",
        "/reset": "Clear conversation history and start fresh",
        "/help": "Show available commands",
    }

    def __init__(self, session: InferenceSession):
        self.session = session

    def handle_command(self, text: str) -> str | None:
        """Handle slash commands. Returns 'quit', 'continue', or None (not a command)."""
        stripped = text.strip().lower()

        if stripped in ("/quit", "/exit"):
            return "quit"

        if stripped == "/reset":
            self.session.reset_cache()
            print("\nConversation cleared.\n")
            return "continue"

        if stripped == "/help":
            print("\nCommands:")
            for cmd, desc in self.COMMANDS.items():
                print(f"  {cmd:12s} {desc}")
            print()
            return "continue"

        if stripped.startswith("/"):
            print(f"\nUnknown command: {stripped}. Type /help for commands.\n")
            return "continue"

        return None  # regular message

    @staticmethod
    def format_stats(tokens: int, elapsed: float) -> str:
        tps = tokens / elapsed if elapsed > 0 else 0
        return f"[{tokens} tokens, {tps:.1f} tok/s, {elapsed:.1f}s]"

    def run(self):
        """Main chat loop."""
        model_name = self.session.model_path.split("/")[-1]
        print(f"\nChat with {model_name} (TurboQuant {'ON' if self.session.use_turboquant else 'OFF'})")
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

            # Generate response
            start = time.perf_counter()
            token_count = 0
            try:
                for chunk in self.session.generate_response(user_input):
                    print(chunk, end="", flush=True)
                    token_count += 1
            except KeyboardInterrupt:
                pass  # allow Ctrl+C to stop generation

            elapsed = time.perf_counter() - start
            print(f"\n{self.format_stats(token_count, elapsed)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with an LLM using TurboQuant KV cache compression",
    )
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--key-bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--value-bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-turboquant", action="store_true")

    args = parser.parse_args()

    session = InferenceSession(
        model_path=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        temp=args.temp,
        use_turboquant=not args.no_turboquant,
    )

    print(f"Loading {args.model}...")
    session.load()
    print("Ready!")

    repl = ChatREPL(session)
    repl.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/test_chat.py -v`
Expected: All 4 tests PASS

**Step 5: Add entry point to pyproject.toml**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
mlx-tq-chat = "mlx_turboquant.chat:main"
```

**Step 6: Commit**

```bash
git add src/mlx_turboquant/chat.py tests/test_chat.py pyproject.toml
git commit -m "feat: add interactive chat REPL with TurboQuant compression"
```

---

## Task 3: OpenAI-Compatible API Server

FastAPI server implementing `POST /v1/chat/completions` with streaming (SSE) support. Any tool that speaks the OpenAI API (Continue, Cursor, Open WebUI, etc.) can connect.

**Files:**
- Create: `src/mlx_turboquant/server.py`
- Test: `tests/test_server.py`
- Modify: `pyproject.toml` (add `mlx-tq-server` entry point + fastapi dependency)

**Step 1: Write the failing test**

```python
# tests/test_server.py
"""Tests for the OpenAI-compatible API server."""

import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_session():
    """Create a mock InferenceSession."""
    from mlx_turboquant._session import InferenceSession

    session = MagicMock(spec=InferenceSession)
    session.model_path = "test-model"
    session.model = MagicMock()
    return session


def test_chat_completions_non_streaming(mock_session):
    """POST /v1/chat/completions returns a complete response."""
    from mlx_turboquant.server import create_app
    from fastapi.testclient import TestClient

    mock_session.generate_response.return_value = iter(["Hello", " world", "!"])
    app = create_app(mock_session)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello world!"
    assert data["object"] == "chat.completion"


def test_chat_completions_streaming(mock_session):
    """POST /v1/chat/completions with stream=true returns SSE events."""
    from mlx_turboquant.server import create_app
    from fastapi.testclient import TestClient

    mock_session.generate_response.return_value = iter(["Hello", " world"])
    app = create_app(mock_session)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 200
    # SSE events contain "data:" lines
    text = resp.text
    assert "data:" in text


def test_models_endpoint(mock_session):
    """GET /v1/models returns the loaded model."""
    from mlx_turboquant.server import create_app
    from fastapi.testclient import TestClient

    app = create_app(mock_session)
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pip install fastapi uvicorn httpx && pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_turboquant.server'`

**Step 3: Write minimal implementation**

```python
# src/mlx_turboquant/server.py
"""
OpenAI-compatible API server with TurboQuant KV cache compression.

Implements:
  POST /v1/chat/completions  (streaming + non-streaming)
  GET  /v1/models

Usage:
    mlx-tq-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000
"""

import argparse
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_turboquant._session import InferenceSession


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False


def create_app(session: InferenceSession) -> FastAPI:
    """Create FastAPI app wired to the given session."""

    app = FastAPI(title="mlx-turboquant", description="TurboQuant LLM Server")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": session.model_path,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        # Use the last user message for generation
        user_msg = req.messages[-1].content
        # Sync full conversation history into session
        session._messages = [m.model_dump() for m in req.messages[:-1]]

        if req.stream:
            return _stream_response(session, user_msg, req)
        return _complete_response(session, user_msg, req)

    return app


def _make_id():
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _complete_response(session: InferenceSession, user_msg: str, req: ChatRequest):
    """Non-streaming: collect full response, return as one JSON object."""
    chunks = list(session.generate_response(user_msg, max_tokens=req.max_tokens))
    full_text = "".join(chunks)

    return {
        "id": _make_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": session.model_path,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(chunks),
            "total_tokens": len(chunks),
        },
    }


def _stream_response(session: InferenceSession, user_msg: str, req: ChatRequest):
    """Streaming: yield SSE events as tokens are generated."""
    completion_id = _make_id()
    created = int(time.time())

    def event_stream():
        for chunk_text in session.generate_response(user_msg, max_tokens=req.max_tokens):
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": session.model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Final event
        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": session.model_path,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant LLM API server")
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--key-bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--value-bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--no-turboquant", action="store_true")

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
    print(f"Ready! Serving on http://{args.host}:{args.port}")

    import uvicorn
    app = create_app(session)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/test_server.py -v`
Expected: All 3 tests PASS

**Step 5: Update pyproject.toml**

Add to `[project.scripts]`:

```toml
mlx-tq-server = "mlx_turboquant.server:main"
```

Add to `[project.optional-dependencies]`:

```toml
server = ["fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]
```

**Step 6: Commit**

```bash
git add src/mlx_turboquant/server.py tests/test_server.py pyproject.toml
git commit -m "feat: add OpenAI-compatible API server with streaming support"
```

---

## Task 4: Wire Into Package and Update __init__.py

Export the new modules and verify everything integrates.

**Files:**
- Modify: `src/mlx_turboquant/__init__.py`

**Step 1: Update __init__.py**

Add to imports and `__all__`:

```python
from mlx_turboquant._session import InferenceSession

__all__ = [
    "TurboQuantCache",
    "patch_model",
    "make_turboquant_cache",
    "generate",
    "InferenceSession",
]
```

**Step 2: Run full test suite**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pytest tests/ -v`
Expected: All tests PASS

**Step 3: Verify entry points install**

Run: `cd /Users/yahavzamari/Projects/GitHub/mlx-turboquant && source venv/bin/activate && pip install -e ".[server]" && mlx-tq-chat --help && mlx-tq-server --help`
Expected: Both commands print help text

**Step 4: Commit**

```bash
git add src/mlx_turboquant/__init__.py
git commit -m "feat: export InferenceSession and register chat/server entry points"
```

---

## Usage After Implementation

### Chat (like `ollama run`)

```bash
# Small model
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit

# 32B model with TurboQuant compression
mlx-tq-chat --model mlx-community/Qwen2.5-32B-Instruct-4bit --key-bits 3

# Without TurboQuant (for comparison)
mlx-tq-chat --model mlx-community/Qwen2.5-3B-Instruct-4bit --no-turboquant
```

### API Server (OpenAI-compatible)

```bash
# Start server
mlx-tq-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000

# Use with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}'

# Use with any OpenAI-compatible client
# Set base_url to http://localhost:8000/v1
```
