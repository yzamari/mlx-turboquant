"""Tests for the OpenAI-compatible API server."""

import json
from unittest.mock import MagicMock
import pytest


def _make_mock_session():
    """Create a mock InferenceSession with predictable generate_response."""
    from mlx_turboquant._session import InferenceSession

    session = MagicMock(spec=InferenceSession)
    session.model_path = "test-model"
    session.model = MagicMock()  # non-None means loaded
    session._messages = []
    return session


def _make_fake_responses(texts):
    """Create list of mock GenerationResponse objects."""
    responses = []
    for i, t in enumerate(texts):
        r = MagicMock()
        r.text = t
        r.generation_tokens = i + 1
        r.prompt_tokens = 10
        r.prompt_tps = 100.0
        r.generation_tps = 50.0
        r.peak_memory = 1.0
        r.finish_reason = "stop" if i == len(texts) - 1 else None
        responses.append(r)
    return responses


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a mock session."""
    from mlx_turboquant.server import create_app
    from fastapi.testclient import TestClient

    session = _make_mock_session()
    session.generate_response.return_value = iter(
        _make_fake_responses(["Hello", " world", "!"])
    )
    app = create_app(session)
    return TestClient(app), session


class TestModelsEndpoint:
    def test_returns_loaded_model(self, client):
        tc, _ = client
        resp = tc.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"


class TestChatCompletionsNonStreaming:
    def test_returns_complete_response(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello world!"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_includes_usage(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        data = resp.json()
        assert "usage" in data
        assert data["usage"]["completion_tokens"] == 3

    def test_passes_messages_to_session(self, client):
        tc, session = client
        tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        session.generate_response.assert_called_once()
        call_args = session.generate_response.call_args
        assert call_args[0][0] == "hi"


class TestChatCompletionsStreaming:
    def test_returns_sse_events(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_sse_contains_data_lines(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        # 3 content chunks + 1 finish + 1 [DONE] = 5
        assert len(lines) >= 4

    def test_sse_ends_with_done(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "data: [DONE]" in resp.text

    def test_sse_chunks_have_delta(self, client):
        tc, _ = client
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        data_lines = [
            l for l in resp.text.split("\n") if l.startswith("data: {")
        ]
        first_chunk = json.loads(data_lines[0][6:])
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "delta" in first_chunk["choices"][0]


class TestChatCompletionsConversationState:
    def test_sets_history_before_generating(self, client):
        tc, session = client
        tc.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And 3+3?"},
                ],
            },
        )
        assert session._messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        session.generate_response.assert_called_once_with(
            "And 3+3?",
            max_tokens=1024,
            temp=0.7,
            top_p=0.9,
        )


class TestHealthEndpoint:
    def test_health_check(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
