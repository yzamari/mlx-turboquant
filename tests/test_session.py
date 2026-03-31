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

    def test_reset_clears_cache(self):
        from mlx_turboquant._session import InferenceSession

        s = InferenceSession(model_path="test-model")
        s._cache = [MagicMock()]
        s.reset()
        assert s._cache is None


def _setup_session_with_mocks(monkeypatch):
    """Helper: create a loaded session with mocked mlx_lm and cache."""
    from mlx_turboquant._session import InferenceSession

    fake_model = _make_fake_model()
    fake_tokenizer = MagicMock()

    # Chat template: wraps messages in markers
    def fake_chat_template(messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>{m['content']}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    fake_tokenizer.apply_chat_template = fake_chat_template
    monkeypatch.setattr("mlx_lm.load", lambda path: (fake_model, fake_tokenizer))

    fake_response = MagicMock()
    fake_response.text = "Hi!"
    fake_response.generation_tokens = 2
    fake_response.prompt_tokens = 5
    fake_response.prompt_tps = 100.0
    fake_response.generation_tps = 50.0
    fake_response.peak_memory = 1.0
    fake_response.finish_reason = "stop"
    monkeypatch.setattr(
        "mlx_lm.stream_generate", lambda *a, **kw: iter([fake_response])
    )

    fake_cache = [MagicMock()]
    monkeypatch.setattr(
        "mlx_turboquant._session.make_turboquant_cache",
        lambda model, **kw: fake_cache,
    )

    s = InferenceSession(model_path="test-model")
    s.load()
    return s, fake_cache


class TestSessionGenerateResponse:
    def test_appends_user_and_assistant_messages(self, monkeypatch):
        s, _ = _setup_session_with_mocks(monkeypatch)
        chunks = list(s.generate_response("hello"))

        assert chunks[0].text == "Hi!"
        assert len(s.messages) == 2
        assert s.messages[0] == {"role": "user", "content": "hello"}
        assert s.messages[1] == {"role": "assistant", "content": "Hi!"}


class TestMultiTurnCacheReuse:
    def test_cache_created_on_first_turn(self, monkeypatch):
        s, fake_cache = _setup_session_with_mocks(monkeypatch)
        assert s._cache is None
        list(s.generate_response("hello"))
        assert s._cache is fake_cache

    def test_cache_reused_on_second_turn(self, monkeypatch):
        s, fake_cache = _setup_session_with_mocks(monkeypatch)
        list(s.generate_response("hello"))
        cache_after_turn1 = s._cache

        # Make make_turboquant_cache raise if called again
        monkeypatch.setattr(
            "mlx_turboquant._session.make_turboquant_cache",
            lambda model, **kw: (_ for _ in ()).throw(
                AssertionError("should not create new cache")
            ),
        )

        list(s.generate_response("how are you?"))
        assert s._cache is cache_after_turn1  # same object

    def test_continuation_tokens_are_suffix(self, monkeypatch):
        s, _ = _setup_session_with_mocks(monkeypatch)

        # Simulate turn 1 completed with known offset
        s._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        cache_mock = MagicMock()
        cache_mock.offset = 10  # pretend 10 tokens already cached
        s._cache = [cache_mock]

        # Mock tokenizer.encode to return known tokens
        full_tokens = list(range(20))  # 20 tokens for full conversation
        s.tokenizer.encode = lambda text: full_tokens

        # Add new user message
        s._messages.append({"role": "user", "content": "follow up"})
        continuation = s._get_continuation_tokens()

        # Should be tokens [10:20] — only the new suffix
        assert continuation == list(range(10, 20))
        assert len(continuation) == 10

    def test_reset_allows_fresh_cache(self, monkeypatch):
        s, _ = _setup_session_with_mocks(monkeypatch)
        list(s.generate_response("hello"))
        assert s._cache is not None

        s.reset()
        assert s._cache is None
        assert s.messages == []

        # Next turn should create fresh cache
        list(s.generate_response("new conversation"))
        assert s._cache is not None
        assert len(s.messages) == 2
