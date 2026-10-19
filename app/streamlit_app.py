"""Interfaz web del asistente turístico (Streamlit).

Para arrancarla (desde la raíz del repositorio):

    streamlit run app/streamlit_app.py

Tiene respuestas en streaming, el estado de las herramientas en tiempo real,
citas con enlace a Google Maps, preguntas de ejemplo, controles para cambiar
los parámetros del modelo y subida de fotos.
"""

import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # para poder importar el paquete aunque no esté instalado

from tenerife_assistant.assistant import build_default_retriever, create_assistant  # noqa: E402
from tenerife_assistant.config import Settings  # noqa: E402

EXAMPLES = [
    "Quiero pasar un día en La Orotava, ¿qué me recomiendas?",
    "¿Merece la pena ir mañana a Benijo a ver el atardecer?",
    "¿Qué tiempo hará el sábado en el Teide?",
    "¿Qué es un guachinche y a cuál voy?",
    "Planifícame una ruta por la Isla Baja",
]

st.set_page_config(page_title="Guía Tenerife · Asistente", page_icon="🌋", layout="wide")


# ---------------------------------------------------------------------------
# Carga de la guía y la configuración
# ---------------------------------------------------------------------------


# cache_resource hace que el índice se cree una sola vez y no en cada recarga de la página.
# (El guion bajo en _settings le dice a Streamlit que no use ese argumento para la caché.)
@st.cache_resource(show_spinner="Cargando la guía y el índice vectorial…")
def load_retriever(_settings: Settings):
    """Crea el buscador una sola vez y lo comparte entre sesiones."""
    return build_default_retriever(_settings)


def get_settings() -> Settings:
    """Lee la configuración o, si falta la API key, enseña un mensaje claro y para la app."""
    try:
        return Settings.from_env()
    except ValueError as exc:
        st.error(f"**No se puede arrancar el asistente.** {exc}")
        st.info("Copia `.env.example` como `.env`, pega tu API key y recarga la página.")
        st.stop()


settings = get_settings()
retriever = load_retriever(settings)

# Cada usuario tiene su propio asistente (con su memoria) guardado en session_state.
if "assistant" not in st.session_state:
    st.session_state.assistant = create_assistant(settings, retriever=retriever)
    st.session_state.messages = []
assistant = st.session_state.assistant


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"Proveedor: **{settings.provider}** · modelo: `{settings.chat.model}`")
    temperature = st.slider("Temperatura", 0.0, 2.0, settings.chat.temperature, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, settings.chat.top_p, 0.05)
    max_tokens = st.slider("Máx. tokens de salida", 256, 4096, settings.chat.max_tokens, 128)
    top_k = st.slider("Fragmentos recuperados (top-k)", 2, 8, settings.top_k)
    streaming = st.toggle("Respuesta en streaming", value=True)
    assistant.settings.chat = replace(
        assistant.settings.chat, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    assistant.settings.top_k = top_k

    st.divider()
    photo = st.file_uploader("📷 Adjunta una foto (opcional)", type=["jpg", "jpeg", "png", "webp"])
    if st.button("🧹 Nueva conversación", use_container_width=True):
        assistant.reset()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Prueba a preguntar")
    clicked = None
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            clicked = example

    st.divider()
    st.caption(
        f"Memoria: {len(assistant.memory.turns)} turnos · {assistant.memory.tokens} / "
        f"{assistant.memory.max_tokens} tokens" + (" · con resumen" if assistant.memory.summary else "")
    )


# ---------------------------------------------------------------------------
# Conversación
# ---------------------------------------------------------------------------

st.title("🌋 Guía Tenerife")
st.caption(
    "Asistente basado en una guía escrita por una persona local. Cita sus fuentes [n] y consulta "
    "la previsión del tiempo en tiempo real (Open-Meteo)."
)


def render_details(sources: list[dict], tools: list[dict], meta: str) -> None:
    """Muestra fuentes citadas, herramientas usadas y métricas del turno."""
    cited = [s for s in sources if s["citada"]]
    if cited:
        with st.expander(f"📚 Fuentes citadas ({len(cited)})"):
            for s in cited:
                link = f" · [Ver en Google Maps]({s['maps']})" if s.get("maps") else ""
                st.markdown(f"**[{s['n']}]** {s['cita']}{link}")
                st.caption(s["texto"][:400] + ("…" if len(s["texto"]) > 400 else ""))
    if tools:
        with st.expander(f"🛠️ Herramientas usadas ({len(tools)})"):
            for t in tools:
                icon = "✅" if t["ok"] else "⚠️"
                st.markdown(f"{icon} `{t['name']}`")
                st.json({"argumentos": t["arguments"], "resultado": t["result"]}, expanded=True)
    if meta:
        st.caption(meta)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=260)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg.get("sources", []), msg.get("tools", []), msg.get("meta", ""))

prompt = st.chat_input("Pregunta lo que quieras sobre Tenerife…") or clicked

if prompt:
    image_bytes = photo.getvalue() if photo else None
    image_mime = photo.type if photo else "image/jpeg"
    st.session_state.messages.append({"role": "user", "content": prompt, "image": image_bytes})
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=260)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Pensando…", expanded=False)
        result: dict = {}

        def token_stream():
            """Recorre los eventos del asistente: el estado va a st.status y el texto a la pantalla."""
            for event in assistant.chat_stream(
                prompt, image=image_bytes, image_mime=image_mime, stream=streaming
            ):
                if event.type == "status":
                    status.update(label=event.text[:90])
                elif event.type == "tool":
                    ok = "✅" if event.data["ok"] else "⚠️"
                    status.write(f"{ok} {event.text}: `{event.data['arguments']}`")
                elif event.type == "token":
                    yield event.text
                elif event.type == "final":
                    result["response"] = event.data

        st.write_stream(token_stream())
        response = result["response"]
        status.update(label="Listo", state="error" if response.error else "complete")
        if response.error:
            st.warning(response.error)
        meta = (
            f"⏱️ {response.latency_s:.1f} s · {response.llm_calls} llamadas al LLM · "
            f"{response.usage.get('total_tokens', 0)} tokens"
        )
        render_details(response.sources, response.tool_calls, meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text,
            "sources": response.sources,
            "tools": response.tool_calls,
            "meta": meta,
        }
    )
