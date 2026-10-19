"""Memoria de la conversación con límite de tokens.

Uso la estrategia "summary buffer":

- Se guardan los turnos completos (pregunta del usuario + respuesta final).
- Si el historial pasa de max_tokens, los turnos más antiguos se resumen con el
  modelo auxiliar y solo los últimos keep_last_turns se guardan literales.
- Si no hay resumidor (o falla), se borran los turnos más antiguos (ventana
  deslizante). Así nunca se supera el límite.

No guardo en memoria el contexto recuperado ni los mensajes de las
herramientas: ocupan muchos tokens y la respuesta final ya resume lo
importante.
"""

import logging

import tiktoken

logger = logging.getLogger(__name__)

# o200k_base es el tokenizador de GPT-4o/4.1. Gemini usa otro, así que con
# Gemini el recuento es aproximado (pero suficiente para controlar el tamaño).
ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Cuenta los tokens de un texto."""
    return len(ENCODING.encode(text or ""))


def count_message_tokens(messages: list[dict]) -> int:
    """Cuenta los tokens aproximados de una lista de mensajes (+4 por mensaje de formato)."""
    total = 0
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, list):  # mensajes con imagen: solo cuento el texto
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        total += count_tokens(content) + 4
    return total


class Turn:
    """Un turno de la conversación: pregunta del usuario y respuesta del asistente."""

    def __init__(self, user: str, assistant: str):
        self.user = user
        self.assistant = assistant

    def as_messages(self) -> list[dict]:
        """Devuelve el turno como mensajes de chat."""
        return [{"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant}]

    @property
    def tokens(self) -> int:
        """Tokens aproximados del turno."""
        return count_message_tokens(self.as_messages())


SUMMARY_PROMPT = (
    "Resume en español, en 3-5 frases, la conversación entre un turista y un asistente "
    "sobre Tenerife. Conserva lugares, fechas, preferencias del usuario y decisiones tomadas. "
    "Si ya existe un resumen previo, intégralo."
)


class ConversationMemory:
    """Historial de la conversación con un máximo de tokens.

    Args:
        max_tokens: máximo de tokens para el resumen + los turnos literales.
        keep_last_turns: turnos recientes que nunca se resumen.
        summarizer: función (resumen_previo, turnos) -> nuevo_resumen.
            Si es None, se usa la ventana deslizante.
    """

    def __init__(self, max_tokens: int = 1500, keep_last_turns: int = 2, summarizer=None):
        self.max_tokens = max_tokens
        self.keep_last_turns = keep_last_turns
        self.summarizer = summarizer
        self.summary = ""
        self.turns: list[Turn] = []
        self.events: list[str] = []  # qué ha pasado con la memoria (para verlo en el notebook)

    @property
    def tokens(self) -> int:
        """Tokens que ocupa ahora mismo la memoria."""
        return count_tokens(self.summary) + sum(t.tokens for t in self.turns)

    def add_turn(self, user: str, assistant: str) -> None:
        """Añade un turno y, si hace falta, recorta la memoria."""
        self.turns.append(Turn(user, assistant))
        self._enforce_budget()

    def as_messages(self) -> list[dict]:
        """Devuelve los turnos literales como mensajes de chat.

        El resumen no va aquí: el asistente lo añade al system prompt, porque
        algunos proveedores no aceptan mensajes "system" a mitad de conversación.
        """
        messages = []
        for turn in self.turns:
            messages.extend(turn.as_messages())
        return messages

    def last_user_messages(self, n: int = 2) -> list[str]:
        """Devuelve las últimas n preguntas del usuario."""
        return [t.user for t in self.turns[-n:]]

    def clear(self) -> None:
        """Vacía la memoria."""
        self.summary = ""
        self.turns.clear()
        self.events.clear()

    def _enforce_budget(self) -> None:
        """Resume o borra turnos antiguos hasta que la memoria quepa en max_tokens."""
        if self.tokens <= self.max_tokens:
            return

        if self.keep_last_turns:
            old = self.turns[: -self.keep_last_turns]
        else:
            old = list(self.turns)
        recent = self.turns[len(old) :]

        # 1) Intento resumir los turnos antiguos.
        if old and self.summarizer is not None:
            try:
                new_summary = (self.summarizer(self.summary, old) or "").strip()
            except Exception as exc:  # si falla el resumen, pasamos a la ventana deslizante
                new_summary = ""
                logger.warning("No se pudo resumir la memoria (%s); uso ventana deslizante", exc)
            if new_summary:
                self.summary = new_summary
                self.turns = recent
                self.events.append(f"resumidos {len(old)} turnos -> {count_tokens(self.summary)} tokens")
                logger.info("Memoria: %s", self.events[-1])

        # 2) Si aun así no cabe, borro los turnos más antiguos (ventana deslizante).
        while self.tokens > self.max_tokens and len(self.turns) > 1:
            self.turns.pop(0)
            self.events.append("descartado el turno más antiguo")
        if self.tokens > self.max_tokens and self.summary:
            self.summary = ""
            self.events.append("descartado el resumen")


def make_llm_summarizer(llm):
    """Crea una función que resume turnos con el modelo auxiliar del LLMClient."""

    def summarize(previous: str, turns: list[Turn]) -> str:
        transcript = "\n".join(f"Usuario: {t.user}\nAsistente: {t.assistant}" for t in turns)
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Resumen previo: {previous or '(ninguno)'}\n\n{transcript}"},
        ]
        return llm.simple(messages, purpose="summary", max_tokens=400)

    return summarize
