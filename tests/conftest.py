"""Fixtures compartidas: la guía real, un embedder falso y un LLM falso.

Los tests no llaman a ninguna API ni descargan el modelo de embeddings, así que
tardan pocos segundos y no necesitan API key (también se ejecutan en la CI de GitHub).
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from openai.types.chat import ChatCompletion

from tenerife_assistant.config import ModelParams, Settings
from tenerife_assistant.ingest import build_chunks, load_guide
from tenerife_assistant.llm import StreamResult
from tenerife_assistant.retrieval import BM25Index, DenseIndex, Retriever, tokenize_es

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "TENERIFE.pdf"
FIXED_TODAY = date(2026, 10, 11)


class HashEmbedder:
    """Embedder falso para los tests: cuenta palabras con un hash (siempre da lo mismo)."""

    name = "hash-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, kind="passage"):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize_es(text):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


def make_completion(content: str | None = None, tool_calls: list[dict] | None = None) -> ChatCompletion:
    """Crea una respuesta (ChatCompletion) del SDK de OpenAI a partir de datos sencillos."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                **({"extra_content": tc["extra_content"]} if "extra_content" in tc else {}),
            }
            for i, tc in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


class FakeLLM:
    """LLM falso: devuelve las respuestas de `script` en orden.

    Guarda los mensajes que recibe para poder comprobarlos en los tests.
    """

    def __init__(self, script: list[ChatCompletion], utility_reply: str = "consulta reescrita"):
        self.script = list(script)
        self.utility_reply = utility_reply
        self.calls: list[list[dict]] = []
        self.history: list = []
        self.settings = None

    def _next(self, messages):
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        self.history.append("call")
        if not self.script:
            raise AssertionError("El LLM falso se ha quedado sin respuestas")
        return self.script.pop(0)

    def complete(self, messages, tools=None, **kwargs):
        return self._next(messages)

    def stream(self, messages, tools=None, params=None):
        response = self._next(messages)
        msg = response.choices[0].message
        if msg.content:
            for word in msg.content.split(" "):
                yield word + " "
        data = {"role": "assistant"}
        if msg.content:
            data["content"] = " ".join(msg.content.split(" ")) + " "
        if msg.tool_calls:
            data["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls]
        yield StreamResult(
            message=data, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

    def simple(self, messages, purpose="utility", max_tokens=512):
        self.history.append(purpose)
        return self.utility_reply


@pytest.fixture(scope="session")
def sections():
    return load_guide(PDF)


@pytest.fixture(scope="session")
def chunks(sections):
    return build_chunks(sections, "structural", max_chars=1000)


@pytest.fixture(scope="session")
def retriever(chunks):
    dense = DenseIndex(HashEmbedder()).build(chunks)
    return Retriever(dense, BM25Index(chunks), method="hybrid")


@pytest.fixture
def settings(tmp_path):
    params = ModelParams(model="fake", temperature=0.0, top_p=1.0, max_tokens=100)
    return Settings(
        provider="gemini",
        api_key="test",
        base_url=None,
        chat=params,
        utility=params,
        weather_mode="mock",
        log_dir=tmp_path / "logs",
        index_dir=tmp_path / "index",
    )
