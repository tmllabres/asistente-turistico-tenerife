"""Conexión con el LLM.

Uso el SDK oficial de OpenAI tanto para OpenAI como para Gemini (Gemini tiene
un endpoint compatible con OpenAI). Además de hacer la llamada, el cliente:

- Reintenta cuando se supera el límite de peticiones por minuto (error 429).
- Cambia a un modelo de reserva si se acaba la cuota diaria del plan gratuito.
- Traduce los errores de la API a mensajes en castellano para el usuario.
- Guarda el modelo, la latencia y los tokens de cada llamada (observabilidad).
- Permite respuestas en streaming.
"""

import logging
import re
import time
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from .config import ModelParams, Settings

logger = logging.getLogger(__name__)

# Algunos modelos abiertos (Gemma) escriben su razonamiento entre <thought> y </thought>.
THOUGHT_RE = re.compile(r"<thought>.*?</thought>\s*", flags=re.DOTALL)


class LLMError(RuntimeError):
    """Error del LLM con un mensaje que se puede enseñar al usuario."""


@dataclass
class CallStats:
    """Datos de una llamada al LLM (para la observabilidad)."""

    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    purpose: str = "chat"


@dataclass
class StreamResult:
    """Lo que queda al terminar una respuesta en streaming."""

    message: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: str = ""


def _friendly_error(exc: Exception, model: str) -> LLMError:
    """Convierte los errores del SDK en mensajes claros en castellano."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMError("La API key no es válida o no tiene permisos. Revisa el fichero .env.")
    if isinstance(exc, openai.NotFoundError):
        return LLMError(f"El modelo '{model}' no existe para este proveedor. Revisa LLM_MODEL en .env.")
    if isinstance(exc, openai.RateLimitError):
        if _is_daily_quota(exc):
            return LLMError(
                f"Se ha agotado la cuota diaria gratuita del modelo '{model}'. Prueba otro modelo "
                "(LLM_MODEL en .env) o espera a que se renueve la cuota."
            )
        return LLMError(
            "Se ha superado el límite de peticiones por minuto de la API. "
            "Espera un minuto y vuelve a intentarlo."
        )
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return LLMError("No hay conexión con el proveedor del LLM. Comprueba tu conexión a internet.")
    if isinstance(exc, openai.BadRequestError):
        return LLMError(f"El proveedor rechazó la petición: {exc.message}")
    return LLMError(f"Error inesperado del LLM: {exc}")


def _is_daily_quota(exc: Exception) -> bool:
    """Comprueba si el error 429 es por la cuota diaria y no por la de por minuto."""
    # Gemini lo indica en el mensaje de error con "PerDay".
    return "PerDay" in str(exc)


def retry_delay(exc: Exception, attempt: int) -> float:
    """Calcula cuántos segundos esperar antes de reintentar tras un error 429.

    Gemini dice en el propio error cuánto hay que esperar ("Please retry in
    26.7s"). Si viene ese dato lo uso (con 1 s de margen); si no, espero cada
    vez más: 5, 10, 20... segundos.
    """
    match = re.search(r"retry in ([\d.]+)s|retryDelay'?:\s*'?(\d+)s", str(exc))
    if match:
        seconds = float(match.group(1) or match.group(2))
        return min(seconds + 1.0, 65.0)
    return min(2.0**attempt * 5, 60.0)


def strip_thoughts(text):
    """Quita los bloques <thought>...</thought> de una respuesta."""
    if not text:
        return text
    return THOUGHT_RE.sub("", text).strip() or None


def usage_to_dict(usage) -> dict:
    """Pasa el objeto `usage` del SDK (tokens usados) a un diccionario."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _empty_tool_call() -> dict:
    return {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}


class LLMClient:
    """Cliente del LLM.

    Args:
        settings: configuración (proveedor, API key y parámetros).
        rate_limit_retries: cuántas veces reintentar tras un error 429 por minuto.
    """

    def __init__(self, settings: Settings, rate_limit_retries: int = 4):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=2,
        )
        self.rate_limit_retries = rate_limit_retries
        self.history: list[CallStats] = []
        self.fallbacks = list(settings.fallback_models)
        # Modelos que se han cambiado por uno de reserva: {modelo_original: modelo_de_reserva}
        self.replacements = {}

    def _switch_to_fallback(self, model: str) -> bool:
        """Cambia `model` por el siguiente modelo de reserva que no se haya usado ya."""
        current = self.replacements.get(model, model)
        for candidate in self.fallbacks:
            if candidate != current and candidate not in self.replacements.values():
                self.replacements[model] = candidate
                logger.warning("Cuota diaria agotada en %s; paso al modelo de reserva %s", current, candidate)
                return True
        return False

    def _build_kwargs(self, params=None, tools=None, temperature=None, max_tokens=None) -> dict:
        """Junta los parámetros del modelo y, si las hay, las herramientas."""
        kwargs = (params or self.settings.chat).as_api_kwargs()
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _request(self, messages: list[dict], kwargs: dict):
        """Hace la petición a la API gestionando los 429 y los modelos de reserva.

        Returns:
            La respuesta de la API y el nombre del modelo que se ha usado al final.
        """
        requested = kwargs["model"]
        attempt = 0
        while True:
            model = self.replacements.get(requested, requested)
            call_kwargs = dict(kwargs, model=model)
            if model != requested:
                call_kwargs.pop("reasoning_effort", None)  # el modelo de reserva puede no admitirlo
            try:
                response = self.client.chat.completions.create(messages=messages, **call_kwargs)
                return response, model
            except openai.RateLimitError as exc:
                if _is_daily_quota(exc):
                    # Esperar no sirve de nada: pruebo con otro modelo.
                    if self._switch_to_fallback(requested):
                        continue
                    raise _friendly_error(exc, model) from exc
                # Límite por minuto: espero lo que pide la API y lo vuelvo a intentar.
                if attempt >= self.rate_limit_retries:
                    raise _friendly_error(exc, model) from exc
                delay = retry_delay(exc, attempt)
                logger.warning("Límite de peticiones por minuto (429). Reintento en %.0f s", delay)
                time.sleep(delay)
                attempt += 1
            except openai.OpenAIError as exc:
                raise _friendly_error(exc, model) from exc

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        params: ModelParams | None = None,
        temperature_override: float | None = None,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ):
        """Llamada normal (sin streaming) al LLM.

        Args:
            messages: historial de mensajes en formato OpenAI.
            tools: herramientas que puede usar el modelo (JSON Schema).
            params: parámetros del modelo (por defecto, los del chat).
            temperature_override: otra temperatura solo para esta llamada.
            max_tokens: otro límite de salida solo para esta llamada.
            purpose: etiqueta para las métricas ("chat", "rewrite", "summary"...).

        Returns:
            La respuesta (ChatCompletion) del SDK de OpenAI.

        Raises:
            LLMError: con un mensaje claro si la llamada falla.
        """
        kwargs = self._build_kwargs(params, tools, temperature_override, max_tokens)
        start = time.perf_counter()
        response, model = self._request(messages, kwargs)
        message = response.choices[0].message
        message.content = strip_thoughts(message.content)
        self._record(model, start, response.usage, purpose)
        return response

    def simple(self, messages: list[dict], purpose: str = "utility", max_tokens: int = 512) -> str:
        """Llamada corta con el modelo auxiliar. Devuelve solo el texto."""
        response = self.complete(
            messages, params=self.settings.utility, max_tokens=max_tokens, purpose=purpose
        )
        return (response.choices[0].message.content or "").strip()

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None, params: ModelParams | None = None
    ):
        """Llamada en streaming.

        Es un generador: va devolviendo trozos de texto (str) según llegan y, al
        final, un StreamResult con el mensaje completo (incluidas las llamadas a
        herramientas) y los tokens usados.
        """
        kwargs = self._build_kwargs(params, tools)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        start = time.perf_counter()
        stream, model = self._request(messages, kwargs)

        content_parts = []
        thought_buffer = ""  # para quitar un posible <thought>...</thought> al principio
        tool_calls = []
        usage = None
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text = delta.content
                    # Si la respuesta empieza con <thought>, acumulo hasta que se cierre y lo quito.
                    if not content_parts and (thought_buffer or text.lstrip().startswith("<thought>")):
                        thought_buffer += text
                        if "</thought>" not in thought_buffer:
                            continue
                        text = strip_thoughts(thought_buffer) or ""
                        thought_buffer = ""
                    if text:
                        content_parts.append(text)
                        yield text
                for tool_call in delta.tool_calls or []:
                    self._accumulate_tool_call(tool_calls, tool_call)
        except openai.OpenAIError as exc:
            raise _friendly_error(exc, model) from exc

        self._record(model, start, usage, "chat-stream")
        message = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        yield StreamResult(message=message, usage=usage_to_dict(usage), model=model)

    @staticmethod
    def _accumulate_tool_call(calls: list[dict], tool_call) -> None:
        """Junta los trozos de las llamadas a herramientas que llegan en streaming.

        OpenAI manda cada llamada en varios trozos y cada trozo trae un `index`
        que dice a qué llamada pertenece. Gemini manda cada llamada entera y sin
        `index`, y además trae `extra_content` (la "thought signature"), que hay
        que devolverle en la siguiente petición o da error.
        """
        data = tool_call.model_dump(exclude_none=True)
        index = data.get("index")
        if index is not None:
            # Estilo OpenAI
            while len(calls) <= index:
                calls.append(_empty_tool_call())
            target = calls[index]
        elif data.get("id") and all(c["id"] != data["id"] for c in calls):
            # Estilo Gemini: un id nuevo es una llamada nueva
            calls.append(_empty_tool_call())
            target = calls[-1]
        elif calls:
            target = calls[-1]
        else:
            return

        if data.get("id"):
            target["id"] = data["id"]
        function = data.get("function", {})
        target["function"]["name"] += function.get("name", "") or ""
        target["function"]["arguments"] += function.get("arguments", "") or ""
        if "extra_content" in data:
            target["extra_content"] = data["extra_content"]

    def _record(self, model: str, start: float, usage, purpose: str) -> None:
        """Guarda las métricas de una llamada en self.history."""
        stats = CallStats(
            model=model, latency_s=time.perf_counter() - start, purpose=purpose, **usage_to_dict(usage)
        )
        self.history.append(stats)
        logger.debug("LLM %s %s %.2fs %s tokens", purpose, model, stats.latency_s, stats.total_tokens)
