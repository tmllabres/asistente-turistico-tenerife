"""Configuración del asistente.

Todo se lee de variables de entorno o del fichero .env (que está en .gitignore,
así que nunca se sube al repositorio). La API key no se imprime nunca:
describe() la muestra enmascarada.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Carpeta raíz del proyecto (dos niveles por encima de este fichero).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Valores por defecto de cada proveedor. Uso el SDK de OpenAI para los dos,
# porque Gemini tiene un endpoint compatible con OpenAI: para cambiar de
# proveedor basta con cambiar LLM_PROVIDER en el .env.
PROVIDER_DEFAULTS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        # Flash-Lite es rápido y tiene más cuota gratuita que Flash (que se
        # queda en unas 20 peticiones al día en el plan gratuito).
        "chat_model": "gemini-3.5-flash-lite",
        "utility_model": "gemini-3.1-flash-lite",
        "fallback_models": ["gemini-3.1-flash-lite", "gemma-4-26b-a4b-it"],
        # Google recomienda no bajar la temperatura de Gemini 3 por debajo de
        # 1.0 (puede entrar en bucles). La creatividad la controlo con el
        # prompt y con top_p.
        "temperature": 1.0,
        "top_p": 0.95,
        # En Gemini, max_tokens también cuenta los tokens de razonamiento.
        "max_tokens": 2048,
        "reasoning_effort": "low",
    },
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "chat_model": "gpt-4.1-mini",
        "utility_model": "gpt-4.1-nano",
        "fallback_models": [],
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 800,
        "reasoning_effort": None,
    },
}


@dataclass
class ModelParams:
    """Parámetros que se envían al LLM en cada llamada.

    Attributes:
        model: nombre del modelo.
        temperature: aleatoriedad de la respuesta (0 = casi determinista).
        top_p: nucleus sampling; solo se eligen tokens hasta sumar esta probabilidad.
        max_tokens: máximo de tokens de salida (en Gemini incluye el razonamiento).
        reasoning_effort: cuánto "piensa" el modelo antes de responder
            (None para no enviarlo).
    """

    model: str
    temperature: float
    top_p: float
    max_tokens: int
    reasoning_effort: str | None = None

    def as_api_kwargs(self) -> dict:
        """Devuelve los parámetros listos para client.chat.completions.create()."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs


@dataclass
class Settings:
    """Toda la configuración de la aplicación."""

    provider: str
    api_key: str
    base_url: str | None
    chat: ModelParams
    utility: ModelParams
    fallback_models: tuple = ()
    embedding_model: str = "intfloat/multilingual-e5-small"
    chunk_strategy: str = "structural"
    chunk_max_chars: int = 1000
    retrieval_method: str = "hybrid"
    top_k: int = 4
    memory_max_tokens: int = 1500
    memory_keep_turns: int = 2
    rewrite_queries: str = "auto"  # auto, always o never
    max_tool_rounds: int = 4
    weather_mode: str = "live"
    request_timeout: float = 60.0
    max_retries: int = 5
    pdf_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "TENERIFE.pdf")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "index")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    @classmethod
    def from_env(cls, env_file=None, allow_missing_key: bool = False) -> "Settings":
        """Crea la configuración a partir de las variables de entorno.

        Args:
            env_file: ruta del .env (por defecto, el de la raíz del proyecto).
            allow_missing_key: no dar error si falta la API key. Lo usa el
                Dockerfile para construir el índice sin tener la key.

        Returns:
            Un objeto Settings.

        Raises:
            ValueError: si el proveedor no existe o falta la API key.
        """
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER debe ser uno de {list(PROVIDER_DEFAULTS)}, no '{provider}'")
        defaults = PROVIDER_DEFAULTS[provider]

        api_key = os.getenv(defaults["api_key_env"], "").strip()
        if not api_key and not allow_missing_key:
            raise ValueError(
                f"Falta la variable {defaults['api_key_env']}. Copia .env.example a .env "
                "y pega tu API key (ver README, sección 'Configuración')."
            )

        reasoning = os.getenv("LLM_REASONING_EFFORT", defaults["reasoning_effort"] or "")
        chat = ModelParams(
            model=os.getenv("LLM_MODEL", defaults["chat_model"]),
            temperature=float(os.getenv("LLM_TEMPERATURE", defaults["temperature"])),
            top_p=float(os.getenv("LLM_TOP_P", defaults["top_p"])),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", defaults["max_tokens"])),
            reasoning_effort=reasoning or None,
        )
        # Modelo auxiliar (más rápido y barato) para reescribir preguntas,
        # resumir la memoria y describir fotos.
        utility = ModelParams(
            model=os.getenv("LLM_UTILITY_MODEL", defaults["utility_model"]),
            temperature=chat.temperature,
            top_p=chat.top_p,
            max_tokens=512,
            reasoning_effort=None,
        )

        # Modelos de reserva por si se acaba la cuota diaria del modelo principal.
        fallbacks = os.getenv("LLM_FALLBACK_MODELS")
        if fallbacks is None:
            fallback_models = tuple(defaults["fallback_models"])
        else:
            fallback_models = tuple(m.strip() for m in fallbacks.split(",") if m.strip())

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", defaults["base_url"]) or None,
            chat=chat,
            utility=utility,
            fallback_models=fallback_models,
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            weather_mode=os.getenv("WEATHER_MODE", "live").lower(),
        )

    def describe(self) -> dict:
        """Devuelve la configuración como diccionario, con la API key oculta."""
        data = asdict(self)
        data["api_key"] = "******** (definida)" if self.api_key else "(no definida)"
        # Las rutas se muestran relativas al proyecto para que se lean mejor.
        for name in ("pdf_path", "index_dir", "log_dir"):
            path = Path(data[name])
            if path.is_relative_to(PROJECT_ROOT):
                data[name] = str(path.relative_to(PROJECT_ROOT))
            else:
                data[name] = str(path)
        return data
