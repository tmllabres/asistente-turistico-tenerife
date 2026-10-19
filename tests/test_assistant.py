import json

from tenerife_assistant.assistant import TourismAssistant, looks_dependent
from tenerife_assistant.llm import LLMError
from tenerife_assistant.tools import WeatherService

from .conftest import FIXED_TODAY, FakeLLM, make_completion


def build(settings, retriever, script, **kwargs):
    llm = FakeLLM(script, **kwargs)
    weather = WeatherService(mode="mock", today=lambda: FIXED_TODAY)
    return TourismAssistant(settings, llm, retriever, weather), llm


def test_single_turn_with_citation(settings, retriever):
    assistant, llm = build(
        settings, retriever, [make_completion("En La Orotava ve la Casa de los Balcones [1].")]
    )
    response = assistant.chat("¿Qué ver en La Orotava?")
    assert response.error is None and response.llm_calls == 1
    assert response.cited_sources and response.cited_sources[0]["n"] == 1
    assert response.invalid_citations == []
    # El contexto recuperado viaja en el último mensaje de usuario.
    assert "CONTEXTO DE LA GUÍA" in llm.calls[0][-1]["content"]
    assert len(assistant.memory.turns) == 1
    assert "[1]" not in assistant.memory.turns[0].assistant  # las citas no pasan a la memoria


def test_tool_call_round_trip_keeps_thought_signature(settings, retriever):
    script = [
        make_completion(
            tool_calls=[
                {
                    "name": "get_weather",
                    "arguments": {"fecha": "2026-10-12", "zona": "anaga"},
                    "extra_content": {"google": {"thought_signature": "firma"}},
                }
            ]
        ),
        make_completion("Mañana en Anaga estará despejado (Open-Meteo)."),
    ]
    assistant, llm = build(settings, retriever, script)
    response = assistant.chat("¿Qué tiempo hará mañana en Benijo?")
    assert [t["name"] for t in response.tool_calls] == ["get_weather"]
    assert response.tool_calls[0]["ok"] is True
    second_call = llm.calls[1]
    assistant_msg = next(m for m in second_call if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant_msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "firma"
    tool_msg = second_call[-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["ok"] is True


def test_tool_error_is_returned_to_model(settings, retriever):
    script = [
        make_completion(tool_calls=[{"name": "get_weather", "arguments": {"fecha": "2026-12-25"}}]),
        make_completion("No hay previsión tan lejana."),
    ]
    assistant, llm = build(settings, retriever, script)
    response = assistant.chat("¿Qué tiempo hará en Navidad?")
    assert response.tool_calls[0]["ok"] is False
    assert json.loads(llm.calls[1][-1]["content"])["tipo_error"] == "fuera_de_rango"


def test_search_tool_adds_new_numbered_sources(settings, retriever):
    script = [
        make_completion(
            tool_calls=[
                {"name": "buscar_en_guia", "arguments": {"consulta": "pizzería Rugantino Garachico", "k": 2}}
            ]
        ),
        make_completion("En Garachico está la Pizzería Rugantino [5]."),
    ]
    assistant, _ = build(settings, retriever, script)
    response = assistant.chat("¿Qué miradores hay en Anaga?")
    numbers = [s["n"] for s in response.sources]
    assert numbers == list(range(1, len(numbers) + 1))  # numeración continua y sin duplicados
    new = [s for s in response.sources if s["origen"] == "buscar_en_guia"]
    assert new and any("Rugantino" in s["texto"] for s in new)
    assert response.invalid_citations == []


def test_invalid_citations_are_detected(settings, retriever):
    assistant, _ = build(settings, retriever, [make_completion("Dato inventado [42].")])
    assert assistant.chat("¿Qué ver en Masca?").invalid_citations == [42]


def test_max_tool_rounds_guard(settings, retriever):
    loop = [make_completion(tool_calls=[{"name": "get_weather", "arguments": {"fecha": "2026-10-12"}}])]
    script = loop * settings.max_tool_rounds + [make_completion("Respuesta final.")]
    assistant, llm = build(settings, retriever, script)
    response = assistant.chat("tiempo")
    assert response.text == "Respuesta final."
    assert len(llm.calls) == settings.max_tool_rounds + 1


def test_streaming_emits_tokens_and_final(settings, retriever):
    assistant, _ = build(settings, retriever, [make_completion("Hola desde Tenerife [1].")])
    events = list(assistant.chat_stream("¿Qué ver en Santa Cruz?"))
    types = [e.type for e in events]
    assert types[0] == "status" and types[-1] == "final"
    assert "".join(e.text for e in events if e.type == "token").strip() == "Hola desde Tenerife [1]."


def test_followup_rewrites_query_with_utility_model(settings, retriever):
    script = [make_completion("La Orotava es preciosa [1]."), make_completion("Come en La Carajita [1].")]
    assistant, llm = build(settings, retriever, script, utility_reply="dónde comer en La Orotava")
    assistant.chat("Quiero pasar un día en La Orotava, ¿qué me recomiendas ver?")
    response = assistant.chat("¿Y dónde como por allí?")
    assert response.search_query == "dónde comer en La Orotava"
    assert "rewrite" in llm.history
    # La segunda llamada lleva el turno anterior como historial.
    assert [m["role"] for m in llm.calls[1]][:3] == ["system", "user", "assistant"]


def test_llm_error_is_reported_and_not_stored(settings, retriever):
    assistant, llm = build(settings, retriever, [])

    def boom(*args, **kwargs):
        raise LLMError("Se ha superado el límite de peticiones")

    llm.complete = boom
    response = assistant.chat("¿Qué ver en Masca?")
    assert response.error and response.text.startswith("⚠️")
    assert assistant.memory.turns == []


def test_turns_are_logged(settings, retriever):
    assistant, _ = build(settings, retriever, [make_completion("Ok [1].")])
    assistant.chat("¿Qué ver en Los Gigantes?")
    record = json.loads((settings.log_dir / "turns.jsonl").read_text().splitlines()[-1])
    assert record["event"] == "turn" and record["cited"] == [1] and record["llm_calls"] == 1


def test_looks_dependent_heuristic():
    assert looks_dependent("¿Y dónde como por allí?")
    assert looks_dependent("¿Y en el Teide?")
    assert not looks_dependent("Quiero pasar un día en La Orotava, ¿qué me recomiendas ver?")
