"""Herramientas que el LLM puede llamar (function calling).

- get_weather(fecha, zona): previsión del tiempo real con la API gratuita de
  Open-Meteo (no necesita API key), o simulada con WEATHER_MODE=mock.
- buscar_en_guia(consulta, k): búsqueda extra en la guía, para que el modelo
  pueda buscar más cuando la pregunta toca varios lugares.

Los argumentos se definen con Pydantic: de ahí sale el JSON Schema que se le
pasa al modelo y la validación de lo que el modelo devuelve. Los errores no
lanzan excepciones: se le devuelven al modelo como {"ok": false, ...} para que
se lo explique al usuario. Cada llamada y cada intento HTTP se guardan en
logs/tool_calls.jsonl.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

CANARY_TZ = ZoneInfo("Atlantic/Canary")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 15  # Open-Meteo da previsión para hoy + 15 días


def today_canary() -> date:
    """Devuelve la fecha de hoy en Canarias (la referencia para "mañana", "el sábado"...)."""
    return datetime.now(CANARY_TZ).date()


# ---------------------------------------------------------------------------
# Log en formato JSONL (una línea JSON por evento)
# ---------------------------------------------------------------------------


class JsonlLogger:
    """Guarda eventos como líneas JSON, fáciles de leer luego con pandas.

    Args:
        path: fichero donde escribir. Si es None, no se guarda nada.
    """

    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data) -> None:
        """Añade un evento con la hora actual."""
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event, **data}
        logger.debug("%s", record)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# get_weather
# ---------------------------------------------------------------------------


class Zona(StrEnum):
    """Zonas de la guía con un clima distinto.

    El norte y el sur de Tenerife pueden tener un tiempo muy diferente el mismo
    día, y el Teide está a más de 2000 m de altitud.
    """

    santa_cruz = "santa_cruz"
    la_laguna = "la_laguna"
    anaga = "anaga"
    la_orotava = "la_orotava"
    puerto_de_la_cruz = "puerto_de_la_cruz"
    isla_baja = "isla_baja"
    teide = "teide"
    costa_adeje = "costa_adeje"
    los_gigantes = "los_gigantes"
    masca = "masca"
    el_medano = "el_medano"
    los_cristianos = "los_cristianos"


# Nombre, latitud y longitud de cada zona.
ZONE_INFO = {
    Zona.santa_cruz: ("Santa Cruz de Tenerife", 28.4682, -16.2546),
    Zona.la_laguna: ("San Cristóbal de La Laguna", 28.4874, -16.3141),
    Zona.anaga: ("Parque Rural de Anaga", 28.5330, -16.2730),
    Zona.la_orotava: ("La Orotava", 28.3906, -16.5233),
    Zona.puerto_de_la_cruz: ("Puerto de la Cruz", 28.4143, -16.5485),
    Zona.isla_baja: ("Isla Baja (Garachico)", 28.3733, -16.7640),
    Zona.teide: ("Parque Nacional del Teide (Las Cañadas)", 28.2236, -16.6266),
    Zona.costa_adeje: ("Costa Adeje", 28.0835, -16.7330),
    Zona.los_gigantes: ("Los Gigantes", 28.2450, -16.8410),
    Zona.masca: ("Masca", 28.3060, -16.8400),
    Zona.el_medano: ("El Médano", 28.0440, -16.5360),
    Zona.los_cristianos: ("Los Cristianos", 28.0510, -16.7170),
}

# Códigos del tiempo de la OMM (WMO) que devuelve Open-Meteo.
WMO_CODES = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nuboso",
    3: "cubierto",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna débil",
    53: "llovizna moderada",
    55: "llovizna intensa",
    61: "lluvia débil",
    63: "lluvia moderada",
    65: "lluvia fuerte",
    71: "nieve débil",
    73: "nieve moderada",
    75: "nieve fuerte",
    80: "chubascos débiles",
    81: "chubascos moderados",
    82: "chubascos fuertes",
    95: "tormenta",
    96: "tormenta con granizo",
    99: "tormenta con granizo fuerte",
}


class GetWeatherArgs(BaseModel):
    """Argumentos de get_weather."""

    model_config = ConfigDict(extra="forbid")  # no se aceptan argumentos que no estén aquí

    fecha: date = Field(
        description=(
            "Fecha de la previsión en formato YYYY-MM-DD. Convierte expresiones como "
            "'mañana' o 'el sábado' a una fecha concreta usando la fecha de hoy."
        )
    )
    zona: Zona = Field(
        default=Zona.santa_cruz,
        description=(
            "Zona de Tenerife. Usa la más cercana al lugar del que se habla: p. ej. Benijo "
            "o Pico del Inglés -> anaga; Garachico o Icod -> isla_baja; Siam Park o Playa "
            "de Troya -> costa_adeje; La Tejita -> el_medano."
        ),
    )


class ToolError(Exception):
    """Error controlado de una herramienta (con un tipo y un mensaje)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class WeatherService:
    """Consulta la previsión del tiempo, con reintentos y log de cada intento.

    Args:
        mode: "live" (Open-Meteo de verdad) o "mock" (datos simulados, siempre
            iguales para la misma entrada; sirve para tests o sin conexión).
        timeout: segundos máximos de cada petición HTTP.
        max_retries: número total de intentos si hay errores de red.
        backoff: segundos de espera antes del segundo intento (luego se duplica).
        log: logger JSONL donde se guardan los intentos.
        base_url: URL de la API (en el notebook la cambio para simular una caída).
        today: función que devuelve la fecha de hoy (en los tests se fija una fecha).
        session: sesión HTTP (en los tests se sustituye por una falsa).
    """

    def __init__(
        self,
        mode: str = "live",
        timeout: float = 8.0,
        max_retries: int = 3,
        backoff: float = 0.8,
        log: JsonlLogger | None = None,
        base_url: str = FORECAST_URL,
        today=today_canary,
        session=None,
    ):
        if mode not in {"live", "mock"}:
            raise ValueError("mode debe ser 'live' o 'mock'")
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.log = log or JsonlLogger(None)
        self.base_url = base_url
        self.today = today
        self.session = session or requests.Session()

    def get_weather(self, fecha, zona=Zona.santa_cruz) -> dict:
        """Devuelve la previsión de un día en una zona de Tenerife.

        Nunca lanza excepciones: si algo falla devuelve
        {"ok": False, "tipo_error": ..., "error": ...}.

        Args:
            fecha: fecha como date o como texto "YYYY-MM-DD".
            zona: zona de Tenerife.

        Returns:
            Diccionario con la previsión o con el error.
        """
        try:
            args = GetWeatherArgs(fecha=fecha, zona=zona)
            self._check_range(args.fecha)
            if self.mode == "mock":
                data = self._mock(args)
            else:
                data = self._live(args)
            return {"ok": True, **data}
        except ValidationError as exc:
            return {"ok": False, "tipo_error": "argumentos_invalidos", "error": validation_message(exc)}
        except ToolError as exc:
            return {"ok": False, "tipo_error": exc.kind, "error": exc.message}

    def _check_range(self, fecha: date) -> None:
        """Comprueba que hay previsión para esa fecha (de hoy a 15 días)."""
        today = self.today()
        if fecha < today:
            raise ToolError(
                "fecha_pasada",
                f"La fecha {fecha.isoformat()} ya ha pasado (hoy es {today.isoformat()}). "
                "Solo tengo previsiones a partir de hoy.",
            )
        limit = today + timedelta(days=MAX_FORECAST_DAYS)
        if fecha > limit:
            raise ToolError(
                "fuera_de_rango",
                f"Solo hay previsión hasta el {limit.isoformat()} ({MAX_FORECAST_DAYS} días). "
                f"Para el {fecha.isoformat()} todavía no existe previsión fiable.",
            )

    def _live(self, args: GetWeatherArgs) -> dict:
        """Consulta Open-Meteo. Si hay errores de red, 5xx o 429, reintenta esperando cada vez más."""
        name, lat, lon = ZONE_INFO[args.zona]
        daily_fields = [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "wind_speed_10m_max",
            "uv_index_max",
            "sunrise",
            "sunset",
        ]
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(daily_fields),
            "timezone": "Atlantic/Canary",
            "start_date": args.fecha.isoformat(),
            "end_date": args.fecha.isoformat(),
        }

        last_error = "desconocido"
        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self.session.get(self.base_url, params=params, timeout=self.timeout)
                latency = round((time.perf_counter() - start) * 1000)

                # 5xx y 429 son errores temporales: los trato como un fallo de red para reintentar.
                if response.status_code >= 500 or response.status_code == 429:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)

                # Otros 4xx: la petición está mal y reintentar no sirve.
                if response.status_code >= 400:
                    reason = response.json().get("reason", response.text[:200])
                    self.log.log(
                        "http_attempt",
                        tool="get_weather",
                        attempt=attempt,
                        status="error",
                        http_status=response.status_code,
                        error=reason,
                        latency_ms=latency,
                    )
                    raise ToolError("peticion_invalida", f"El servicio rechazó la petición: {reason}")

                daily = response.json()["daily"]
                self.log.log(
                    "http_attempt",
                    tool="get_weather",
                    attempt=attempt,
                    status="ok",
                    http_status=200,
                    latency_ms=latency,
                    zona=args.zona.value,
                    fecha=args.fecha.isoformat(),
                )
                return self._format(args, name, daily)

            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                latency = round((time.perf_counter() - start) * 1000)
                last_error = f"{type(exc).__name__}: {str(exc)[:120]}" if str(exc) else type(exc).__name__
                self.log.log(
                    "http_attempt",
                    tool="get_weather",
                    attempt=attempt,
                    status="error",
                    error=last_error[:200],
                    latency_ms=latency,
                )
                logger.warning("get_weather intento %d/%d fallido: %s", attempt, self.max_retries, last_error)
                if attempt < self.max_retries:
                    time.sleep(self.backoff * 2 ** (attempt - 1))  # 0.8 s, 1.6 s, ...

            except (KeyError, ValueError) as exc:
                # La API ha respondido algo que no tiene el formato esperado.
                self.log.log(
                    "http_attempt",
                    tool="get_weather",
                    attempt=attempt,
                    status="error",
                    error=f"respuesta inesperada: {exc}",
                )
                raise ToolError(
                    "respuesta_invalida", "El servicio del tiempo devolvió datos inesperados."
                ) from exc

        raise ToolError(
            "servicio_no_disponible",
            f"No he podido contactar con el servicio del tiempo tras {self.max_retries} intentos "
            f"({last_error}). Inténtalo de nuevo en unos minutos.",
        )

    @staticmethod
    def _format(args: GetWeatherArgs, name: str, daily: dict) -> dict:
        """Convierte la respuesta de Open-Meteo en un diccionario sencillo."""

        def first(key):
            # Open-Meteo devuelve una lista por campo; como pido un solo día, cojo el primero.
            values = daily.get(key) or [None]
            return values[0]

        code = first("weather_code")
        sunrise, sunset = first("sunrise"), first("sunset")
        return {
            "fecha": args.fecha.isoformat(),
            "zona": name,
            "estado": WMO_CODES.get(code, f"código {code}"),
            "temp_max_c": first("temperature_2m_max"),
            "temp_min_c": first("temperature_2m_min"),
            "prob_lluvia_pct": first("precipitation_probability_max"),
            "lluvia_mm": first("precipitation_sum"),
            "viento_max_kmh": first("wind_speed_10m_max"),
            "indice_uv": first("uv_index_max"),
            "amanecer": sunrise[-5:] if sunrise else None,
            "atardecer": sunset[-5:] if sunset else None,
            "fuente": "Open-Meteo (open-meteo.com)",
        }

    @staticmethod
    def _mock(args: GetWeatherArgs) -> dict:
        """Previsión inventada pero siempre igual para la misma fecha y zona."""
        # Uso un hash de la fecha y la zona como "semilla" para que el resultado no cambie.
        seed = int(hashlib.md5(f"{args.fecha}{args.zona.value}".encode()).hexdigest(), 16)
        name = ZONE_INFO[args.zona][0]
        altitude_penalty = 12 if args.zona == Zona.teide else 0  # en el Teide hace más frío
        tmax = 22 + seed % 7 - altitude_penalty
        code = [0, 1, 2, 3, 61, 80][seed % 6]
        return {
            "fecha": args.fecha.isoformat(),
            "zona": name,
            "estado": WMO_CODES[code],
            "temp_max_c": tmax,
            "temp_min_c": tmax - 5 - seed % 3,
            "prob_lluvia_pct": 70 if code >= 61 else (seed % 20),
            "lluvia_mm": 3.2 if code >= 61 else 0.0,
            "viento_max_kmh": 10 + seed % 25,
            "indice_uv": 4 + seed % 5,
            "amanecer": "07:58",
            "atardecer": "19:21",
            "fuente": "simulado (WEATHER_MODE=mock)",
        }


def validation_message(exc: ValidationError) -> str:
    """Convierte un error de validación de Pydantic en un mensaje corto."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "argumentos"
        parts.append(f"{loc}: {err['msg']}")
    return "Argumentos no válidos -> " + "; ".join(parts)


# ---------------------------------------------------------------------------
# buscar_en_guia
# ---------------------------------------------------------------------------


class SearchGuideArgs(BaseModel):
    """Argumentos de buscar_en_guia."""

    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(
        min_length=3, description="Qué buscar en la guía, en español y con el nombre del lugar."
    )
    k: int = Field(default=3, ge=1, le=6, description="Número de fragmentos a devolver (1-6).")


# ---------------------------------------------------------------------------
# Definición y ejecución de las herramientas
# ---------------------------------------------------------------------------


def pydantic_to_parameters(model) -> dict:
    """Genera el JSON Schema de un modelo Pydantic para pasárselo al LLM.

    Pydantic mete los enumerados en "$defs" y los referencia con "$ref", y
    añade "title" y "format" a cada campo. Algunos proveedores no aceptan
    "$ref" ni "format: date", así que sustituyo las referencias por su
    contenido y quito lo que sobra.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        # Recorre el esquema y cambia cada {"$ref": ...} por la definición a la que apunta.
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items() if k not in {"title", "format"}}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    params = resolve(schema)
    params.pop("description", None)
    params["additionalProperties"] = False
    return params


@dataclass
class Tool:
    """Una herramienta que puede llamar el LLM."""

    name: str
    description: str
    args_model: type[BaseModel]  # modelo Pydantic con los argumentos
    fn: Callable  # función que ejecuta la herramienta

    def schema(self) -> dict:
        """Devuelve la definición de la herramienta en el formato de la API de chat."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": pydantic_to_parameters(self.args_model),
            },
        }


@dataclass
class ToolResult:
    """Resultado de ejecutar una herramienta."""

    name: str
    arguments: dict | str
    ok: bool
    content: dict
    latency_ms: float
    call_id: str = ""

    def to_message(self) -> dict:
        """Devuelve el mensaje "tool" que se le manda al LLM con el resultado."""
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": json.dumps(self.content, ensure_ascii=False, default=str),
        }


class ToolRegistry:
    """Herramientas disponibles y su ejecución.

    Args:
        log: logger JSONL donde se guarda cada llamada.
    """

    def __init__(self, log: JsonlLogger | None = None):
        self.log = log or JsonlLogger(None)
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Añade una herramienta."""
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """Devuelve las definiciones de todas las herramientas para la API."""
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, raw_arguments, call_id: str = "") -> ToolResult:
        """Valida los argumentos que propone el LLM y ejecuta la herramienta.

        Si algo falla (herramienta que no existe, JSON mal formado, argumentos
        no válidos o un error dentro de la herramienta) no se lanza la
        excepción: se devuelve un resultado con ok=False para que el modelo
        pueda reaccionar.
        """
        start = time.perf_counter()
        arguments = raw_arguments
        try:
            if name not in self.tools:
                raise ToolError("herramienta_desconocida", f"No existe la herramienta '{name}'.")
            tool = self.tools[name]
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ToolError("json_invalido", f"Los argumentos no son JSON válido: {exc}") from exc
            validated = tool.args_model.model_validate(arguments)
            content = tool.fn(**validated.model_dump())
        except ValidationError as exc:
            content = {"ok": False, "tipo_error": "argumentos_invalidos", "error": validation_message(exc)}
        except ToolError as exc:
            content = {"ok": False, "tipo_error": exc.kind, "error": exc.message}
        except Exception as exc:  # un fallo en una herramienta no debe romper la conversación
            logger.exception("Fallo inesperado en la herramienta %s", name)
            content = {"ok": False, "tipo_error": "error_interno", "error": str(exc)}

        ok = bool(content.get("ok", True))
        latency = round((time.perf_counter() - start) * 1000, 1)
        self.log.log(
            "tool_call",
            tool=name,
            arguments=arguments,
            ok=ok,
            error=None if ok else content.get("error"),
            latency_ms=latency,
        )
        return ToolResult(name, arguments, ok, content, latency, call_id)


def make_weather_tool(service: WeatherService) -> Tool:
    """Crea la herramienta get_weather a partir de un WeatherService."""
    return Tool(
        name="get_weather",
        description=(
            "Devuelve la previsión del tiempo en Tenerife para una fecha (de hoy a 15 días) y "
            "una zona: estado del cielo, temperaturas, probabilidad de lluvia, viento, índice UV "
            "y horas de amanecer y atardecer. Úsala siempre que el usuario pregunte por el tiempo "
            "o cuando la recomendación dependa de él (playas, atardeceres, subir al Teide)."
        ),
        args_model=GetWeatherArgs,
        fn=service.get_weather,
    )


def make_search_tool(search_fn) -> Tool:
    """Crea la herramienta buscar_en_guia a partir de una función de búsqueda."""

    def run(consulta: str, k: int = 3) -> dict:
        results = search_fn(consulta, k)
        return {"ok": True, "resultados": results}

    return Tool(
        name="buscar_en_guia",
        description=(
            "Busca fragmentos en la guía turística de Tenerife. Úsala si el contexto que ya tienes "
            "no basta, por ejemplo cuando la pregunta compara varios lugares o cambia de tema. "
            "Cada resultado trae un número de fuente [n] que debes citar."
        ),
        args_model=SearchGuideArgs,
        fn=run,
    )
