"""
OpenAI-compatible API server with TurboQuant KV cache compression.

Endpoints:
    POST /v1/chat/completions  — streaming (SSE) and non-streaming
    GET  /v1/models            — list loaded model
    GET  /health               — health check

Usage:
    mlx-tq-server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8000

    # Then connect with any OpenAI-compatible client:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}]}'
"""

import argparse
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_turboquant._session import InferenceSession


# --- Request/Response models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


# --- App factory ---


def create_app(session: InferenceSession) -> FastAPI:
    """Create a FastAPI app wired to the given InferenceSession."""

    app = FastAPI(title="mlx-turboquant", description="TurboQuant LLM Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": session.model_path}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": session.model_path,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        # Set conversation history (all messages except the last user message)
        user_msg = req.messages[-1].content
        session._messages = [m.model_dump() for m in req.messages[:-1]]

        if req.stream:
            return _stream_response(session, user_msg, req)
        return _complete_response(session, user_msg, req)

    return app


# --- Response builders ---


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _complete_response(
    session: InferenceSession, user_msg: str, req: ChatRequest
):
    """Collect full generation, return single JSON response."""
    full_text = ""
    last_response = None
    for response in session.generate_response(
        user_msg,
        max_tokens=req.max_tokens,
        temp=req.temperature,
        top_p=req.top_p,
    ):
        full_text += response.text
        last_response = response

    prompt_tokens = last_response.prompt_tokens if last_response else 0
    gen_tokens = last_response.generation_tokens if last_response else 0

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
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
        },
    }


def _stream_response(
    session: InferenceSession, user_msg: str, req: ChatRequest
):
    """Return SSE stream of chat completion chunks."""
    completion_id = _make_id()
    created = int(time.time())

    def event_stream():
        for response in session.generate_response(
            user_msg,
            max_tokens=req.max_tokens,
            temp=req.temperature,
            top_p=req.top_p,
        ):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": session.model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": response.text},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        done_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": session.model_path,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- CLI entry point ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurboQuant LLM API server")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--key-bits", type=int, default=3, choices=[2, 3, 4]
    )
    parser.add_argument(
        "--value-bits", type=int, default=2, choices=[2, 4]
    )
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--no-turboquant", action="store_true")
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
    print(f"Ready! http://{args.host}:{args.port}")

    import uvicorn

    app = create_app(session)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
