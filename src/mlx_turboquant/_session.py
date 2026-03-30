"""
Shared inference session: loads model once, manages conversation state.

Used by both the chat REPL and the API server to avoid duplicating
model loading and cache management logic.
"""

import threading
from typing import Generator

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from mlx_turboquant.patch import make_turboquant_cache


class InferenceSession:
    """Holds a loaded model, tokenizer, conversation history, and TQ config.

    Thread-safe: a lock serializes generation calls so MLX operations
    don't conflict when the API server handles concurrent requests.
    """

    def __init__(
        self,
        model_path: str,
        key_bits: int = 3,
        value_bits: int = 2,
        buffer_size: int = 128,
        value_group_size: int = 32,
        use_turboquant: bool = True,
    ):
        self.model_path = model_path
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.value_group_size = value_group_size
        self.use_turboquant = use_turboquant

        self.model = None
        self.tokenizer = None
        self._messages: list[dict] = []
        self._lock = threading.Lock()

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def load(self):
        """Load model and tokenizer from model_path. Call once before generating."""
        if self.model is not None:
            raise RuntimeError("Model already loaded")
        self.model, self.tokenizer = mlx_lm.load(self.model_path)

    def reset(self):
        """Clear conversation history to start a fresh chat."""
        self._messages = []

    def _build_prompt(self, messages: list[dict]) -> str:
        """Format messages list into a string using the tokenizer's chat template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: concatenate content
        return "\n".join(m["content"] for m in messages)

    def _make_cache(self):
        """Create a fresh TurboQuantCache list for this generation."""
        if not self.use_turboquant:
            return None
        return make_turboquant_cache(
            self.model,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            value_group_size=self.value_group_size,
            buffer_size=self.buffer_size,
        )

    def generate_response(
        self,
        user_message: str,
        max_tokens: int = 1024,
        temp: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator:
        """Add user message, generate streaming response, update history.

        Yields GenerationResponse objects from mlx_lm.stream_generate.
        After the generator is exhausted, the assistant message is appended
        to self._messages.

        Thread-safe via self._lock.
        """
        with self._lock:
            self._messages.append({"role": "user", "content": user_message})
            formatted = self._build_prompt(self._messages)
            cache = self._make_cache()

            sampler = make_sampler(temp=temp, top_p=top_p)
            gen_kwargs = dict(max_tokens=max_tokens, sampler=sampler)
            if cache is not None:
                gen_kwargs["prompt_cache"] = cache

            full_text = ""
            for response in mlx_lm.stream_generate(
                self.model, self.tokenizer, formatted, **gen_kwargs
            ):
                full_text += response.text
                yield response

            self._messages.append({"role": "assistant", "content": full_text})
