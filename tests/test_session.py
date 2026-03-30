"""Tests for the shared inference session."""

from unittest.mock import MagicMock
import pytest


def _make_fake_model(n_layers=4, head_dim=64):
    """Create a mock model that passes make_turboquant_cache's detection."""
    model = MagicMock()
    layers = []
    for _ in range(n_layers):
        layer = MagicMock()
        layer.self_attn = MagicMock()
        layer.self_attn.head_dim = head_dim
        layer.is_linear = False
        layers.append(layer)
    model.layers = layers
    return model


class TestSessionInit:
    def test_stores_config(self):
        from mlx_turboquant._session import InferenceSession

        s = InferenceSession(
            model_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
            key_bits=3,
            value_bits=2,
            buffer_size=128,
        )
        assert s.model_path == "mlx-community/Qwen2.5-3B-Instruct-4bit"
        assert s.key_bits == 3
        assert s.value_bits == 2
        assert s.buffer_size == 128

    def test_model_not_loaded_initially(self):
        from mlx_turboquant._session import InferenceSession

        s = InferenceSession(model_path="test-model")
        assert s.model is None
        assert s.tokenizer is None

    def test_messages_empty_initially(self):
        from mlx_turboquant._session import InferenceSession

        s = InferenceSession(model_path="test-model")
        assert s.messages == []


class TestSessionLoad:
    def test_load_sets_model_and_tokenizer(self, monkeypatch):
        from mlx_turboquant._session import InferenceSession

        fake_model = _make_fake_model()
        fake_tokenizer = MagicMock()
        monkeypatch.setattr("mlx_lm.load", lambda path: (fake_model, fake_tokenizer))

        s = InferenceSession(model_path="test-model")
        s.load()
        assert s.model is fake_model
        assert s.tokenizer is fake_tokenizer

    def test_load_raises_if_already_loaded(self, monkeypatch):
        from mlx_turboquant._session import InferenceSession

        fake_model = _make_fake_model()
        monkeypatch.setattr(
            "mlx_lm.load", lambda path: (fake_model, MagicMock())
        )

        s = InferenceSession(model_path="test-model")
        s.load()
        with pytest.raises(RuntimeError, match="already loaded"):
            s.load()


class TestSessionReset:
    def test_reset_clears_messages(self):
        from mlx_turboquant._session import InferenceSession

        s = InferenceSession(model_path="test-model")
        s._messages = [{"role": "user", "content": "hello"}]
        s.reset()
        assert s.messages == []


class TestSessionGenerateResponse:
    def test_appends_user_and_assistant_messages(self, monkeypatch):
        from mlx_turboquant._session import InferenceSession

        fake_model = _make_fake_model()
        fake_tokenizer = MagicMock()
        fake_tokenizer.apply_chat_template.return_value = "<formatted>"
        monkeypatch.setattr(
            "mlx_lm.load", lambda path: (fake_model, fake_tokenizer)
        )

        # Mock stream_generate to yield fake responses
        fake_response = MagicMock()
        fake_response.text = "Hi there!"
        fake_response.generation_tokens = 3
        fake_response.prompt_tokens = 5
        fake_response.prompt_tps = 100.0
        fake_response.generation_tps = 50.0
        fake_response.peak_memory = 1.5
        fake_response.finish_reason = "stop"
        monkeypatch.setattr(
            "mlx_lm.stream_generate", lambda *a, **kw: iter([fake_response])
        )

        # Mock make_turboquant_cache
        monkeypatch.setattr(
            "mlx_turboquant._session.make_turboquant_cache",
            lambda model, **kw: [MagicMock()],
        )

        s = InferenceSession(model_path="test-model")
        s.load()
        chunks = list(s.generate_response("hello"))

        assert chunks[0].text == "Hi there!"
        assert len(s.messages) == 2
        assert s.messages[0] == {"role": "user", "content": "hello"}
        assert s.messages[1] == {"role": "assistant", "content": "Hi there!"}
