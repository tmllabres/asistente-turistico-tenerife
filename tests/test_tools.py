import json
from datetime import timedelta

import requests

from tenerife_assistant.tools import (
    GetWeatherArgs,
    JsonlLogger,
    ToolRegistry,
    WeatherService,
    make_weather_tool,
    pydantic_to_parameters,
)

from .conftest import FIXED_TODAY


def service(**kwargs):
    return WeatherService(today=lambda: FIXED_TODAY, backoff=0, **kwargs)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Sesión HTTP que devuelve (o lanza) lo que le digamos, en orden."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = 0

    def get(self, url, params=None, timeout=None):
        self.requests += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


DAILY_OK = {
    "daily": {
        "weather_code": [2],
        "temperature_2m_max": [24.1],
        "temperature_2m_min": [18.0],
        "precipitation_probability_max": [10],
        "precipitation_sum": [0.0],
        "wind_speed_10m_max": [15.2],
        "uv_index_max": [6.0],
        "sunrise": ["2026-10-12T07:59"],
        "sunset": ["2026-10-12T19:20"],
    }
}


def test_schema_is_flat_and_strict():
    params = pydantic_to_parameters(GetWeatherArgs)
    text = json.dumps(params)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    assert params["required"] == ["fecha"]
    assert "teide" in params["properties"]["zona"]["enum"]
    assert params["additionalProperties"] is False


def test_mock_is_deterministic():
    s = service(mode="mock")
    day = (FIXED_TODAY + timedelta(days=2)).isoformat()
    first, second = s.get_weather(day, "teide"), s.get_weather(day, "teide")
    assert first == second and first["ok"] and first["zona"].startswith("Parque Nacional del Teide")


def test_past_and_far_dates_are_rejected():
    s = service(mode="mock")
    past = s.get_weather((FIXED_TODAY - timedelta(days=1)).isoformat())
    far = s.get_weather((FIXED_TODAY + timedelta(days=40)).isoformat())
    assert past["tipo_error"] == "fecha_pasada"
    assert far["tipo_error"] == "fuera_de_rango"


def test_invalid_arguments():
    s = service(mode="mock")
    assert s.get_weather("2026-02-30")["tipo_error"] == "argumentos_invalidos"
    assert s.get_weather("2026-10-12", "lanzarote")["tipo_error"] == "argumentos_invalidos"


def test_live_success_formats_response(tmp_path):
    session = FakeSession([FakeResponse(200, DAILY_OK)])
    result = service(session=session, log=JsonlLogger(tmp_path / "log.jsonl")).get_weather(
        "2026-10-12", "anaga"
    )
    assert result["ok"] and result["estado"] == "parcialmente nuboso"
    assert result["atardecer"] == "19:20" and result["zona"] == "Parque Rural de Anaga"


def test_live_retries_then_succeeds(tmp_path):
    log_path = tmp_path / "log.jsonl"
    session = FakeSession([requests.Timeout("lento"), FakeResponse(503, {}), FakeResponse(200, DAILY_OK)])
    result = service(session=session, log=JsonlLogger(log_path)).get_weather("2026-10-12")
    assert result["ok"] and session.requests == 3
    attempts = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [a["status"] for a in attempts] == ["error", "error", "ok"]


def test_live_gives_up_after_max_retries():
    session = FakeSession([requests.ConnectionError("sin red")] * 3)
    result = service(session=session, max_retries=3).get_weather("2026-10-12")
    assert result == {
        "ok": False,
        "tipo_error": "servicio_no_disponible",
        "error": result["error"],
    }
    assert "3 intentos" in result["error"] and session.requests == 3


def test_client_errors_are_not_retried():
    session = FakeSession([FakeResponse(400, {"reason": "Parameter error"})])
    result = service(session=session).get_weather("2026-10-12")
    assert result["tipo_error"] == "peticion_invalida" and session.requests == 1


def test_registry_handles_bad_calls(tmp_path):
    registry = ToolRegistry(log=JsonlLogger(tmp_path / "calls.jsonl"))
    registry.register(make_weather_tool(service(mode="mock")))
    assert registry.execute("no_existe", "{}").content["tipo_error"] == "herramienta_desconocida"
    assert registry.execute("get_weather", "{fecha:").content["tipo_error"] == "json_invalido"
    extra = registry.execute("get_weather", json.dumps({"fecha": "2026-10-12", "ciudad": "x"}))
    assert extra.content["tipo_error"] == "argumentos_invalidos"
    ok = registry.execute("get_weather", json.dumps({"fecha": "2026-10-12", "zona": "teide"}), call_id="c1")
    assert ok.ok and ok.to_message()["tool_call_id"] == "c1"
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 4
