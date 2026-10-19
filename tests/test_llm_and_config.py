import httpx
import openai
import pytest
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from tenerife_assistant.config import Settings
from tenerife_assistant.llm import LLMClient, LLMError, retry_delay, strip_thoughts

from .conftest import make_completion


def _rate_limit(message: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
    return openai.RateLimitError(message, response=response, body=None)


def test_accumulate_openai_style_deltas():
    calls: list[dict] = []
    parts = [
        {"index": 0, "id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"fe'}},
        {"index": 0, "function": {"arguments": 'cha": "2026-10-12"}'}},
    ]
    for p in parts:
        LLMClient._accumulate_tool_call(calls, ChoiceDeltaToolCall.model_validate(p))
    assert calls == [
        {
            "id": "a",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"fecha": "2026-10-12"}'},
        }
    ]


def test_accumulate_gemini_style_deltas_without_index():
    calls: list[dict] = []
    for i, day in enumerate(["12", "13"]):
        delta = {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "get_weather", "arguments": f'{{"fecha": "2026-10-{day}"}}'},
            "extra_content": {"google": {"thought_signature": "sig"}},
        }
        # Gemini no envía "index"; el SDK construye el objeto sin validar (model_construct).
        LLMClient._accumulate_tool_call(calls, ChoiceDeltaToolCall.model_construct(**delta))
    assert [c["id"] for c in calls] == ["call_0", "call_1"]
    assert calls[0]["extra_content"]["google"]["thought_signature"] == "sig"


def test_strip_thoughts():
    assert strip_thoughts("<thought>pienso…</thought>\nHola") == "Hola"
    assert strip_thoughts("Sin pensamientos") == "Sin pensamientos"
    assert strip_thoughts(None) is None


def test_retry_delay_uses_api_hint():
    assert retry_delay(_rate_limit("Please retry in 26.7s."), 0) == pytest.approx(27.7)
    assert retry_delay(_rate_limit("sin pista"), 1) == 10


def test_daily_quota_switches_to_fallback_model(settings):
    settings.fallback_models = ("modelo-reserva",)
    client = LLMClient(settings)
    used = []

    def create(messages, **kwargs):
        used.append(kwargs["model"])
        if kwargs["model"] == "fake":
            raise _rate_limit("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return make_completion("hola")

    client.client.chat.completions.create = create
    response = client.complete([{"role": "user", "content": "hola"}])
    assert response.choices[0].message.content == "hola"
    assert used == ["fake", "modelo-reserva"]
    assert client.history[-1].model == "modelo-reserva"


def test_daily_quota_without_fallback_gives_friendly_error(settings):
    client = LLMClient(settings)

    def create(messages, **kwargs):
        raise _rate_limit("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    client.client.chat.completions.create = create
    with pytest.raises(LLMError, match="cuota diaria"):
        client.complete([{"role": "user", "content": "hola"}])


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secreta-123456")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    settings = Settings.from_env(env_file=tmp_path / "no-existe.env")
    assert settings.chat.model == "gpt-4.1-mini" and settings.chat.temperature == 0.2
    assert "secreta" not in str(settings.describe())
    assert settings.chat.as_api_kwargs()["max_tokens"] == 800


def test_missing_key_has_clear_message(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings.from_env(env_file=tmp_path / "no-existe.env")
