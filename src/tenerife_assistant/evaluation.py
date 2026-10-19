"""Evaluación reproducible del asistente.

Hay dos niveles:

1. Recuperación (sin LLM, siempre da lo mismo): Hit@k y MRR con preguntas
   que llevan una frase literal de la guía como evidencia. Sirve para
   comparar chunkings, modelos de embeddings y métodos de búsqueda.
2. Extremo a extremo (con LLM): se pasa un conjunto fijo de casos (preguntas
   sueltas y conversaciones) y se comprueba si cita, si usa la herramienta
   correcta, si rechaza lo que no es de la guía, si dice los datos clave, la
   latencia y los tokens. Opcionalmente, otro LLM hace de juez y puntúa si la
   respuesta es fiel al contexto.
"""

import json
import re
import time
import unicodedata

import pandas as pd

from .ingest import build_chunks
from .retrieval import BM25Index, DenseIndex, Retriever

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Pasa a minúsculas y quita las tildes, para comparar textos."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_json(path) -> dict:
    """Lee un fichero JSON en UTF-8."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Evaluación de la recuperación
# ---------------------------------------------------------------------------


def evaluate_retrieval(retriever: Retriever, questions: list[dict], k_values=(1, 3, 5)) -> pd.DataFrame:
    """Calcula, para cada pregunta, en qué posición aparece el primer chunk correcto.

    Un chunk es correcto si contiene alguna de las "evidencias" de la pregunta
    (sin tener en cuenta tildes ni mayúsculas).

    Args:
        retriever: buscador a evaluar.
        questions: preguntas con "id", "pregunta", "tipo" y "evidencias".
        k_values: valores de k para calcular Hit@k.

    Returns:
        DataFrame con una fila por pregunta: "rank" (None si no aparece en los
        max(k_values) primeros), "rr" (1 / rank) y una columna "hit@k" por cada k.
    """
    max_k = max(k_values)
    rows = []
    for q in questions:
        evidences = [normalize(e) for e in q["evidencias"]]
        start = time.perf_counter()
        results = retriever.search(q["pregunta"], k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        rank = next(
            (r.rank for r in results if any(e in normalize(r.chunk.text) for e in evidences)),
            None,
        )
        row = {
            "id": q["id"],
            "tipo": q.get("tipo", ""),
            "pregunta": q["pregunta"],
            "rank": rank,
            "rr": 1.0 / rank if rank else 0.0,
            "latency_ms": latency_ms,
            "top1": results[0].chunk.header if results else "",
        }
        for k in k_values:
            row[f"hit@{k}"] = float(rank is not None and rank <= k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_retrieval(df: pd.DataFrame, label: dict | None = None) -> dict:
    """Calcula la media de las métricas de evaluate_retrieval()."""
    summary = dict(label or {})
    for col in df.columns:
        if col.startswith("hit@"):
            summary[col] = round(df[col].mean(), 3)
    summary["MRR"] = round(df["rr"].mean(), 3)
    summary["latencia_ms"] = round(df["latency_ms"].mean(), 2)
    return summary


def retrieval_grid(
    sections,
    questions: list[dict],
    embedder,
    chunk_configs: list,
    methods=("dense", "bm25", "hybrid"),
    k_values=(1, 3, 5),
) -> pd.DataFrame:
    """Evalúa todas las combinaciones de chunking y método de búsqueda.

    Args:
        sections: lugares de la guía.
        questions: preguntas con su evidencia.
        embedder: modelo de embeddings (el mismo para todas las combinaciones).
        chunk_configs: lista de (estrategia, parámetros).
        methods: métodos de búsqueda a comparar.
        k_values: valores de k para Hit@k.

    Returns:
        Una fila por combinación con Hit@k, MRR, nº de chunks y tamaño medio.
    """
    rows = []
    for strategy, params in chunk_configs:
        chunks = build_chunks(sections, strategy, **params)
        start = time.perf_counter()
        dense = DenseIndex(embedder).build(chunks)
        build_s = time.perf_counter() - start
        bm25 = BM25Index(chunks)
        label = f"{strategy}-" + "-".join(str(v) for v in params.values())
        for method in methods:
            df = evaluate_retrieval(Retriever(dense, bm25, method=method), questions, k_values)
            rows.append(
                summarize_retrieval(
                    df,
                    {
                        "chunking": label,
                        "metodo": method,
                        "n_chunks": len(chunks),
                        "chars_medio": round(sum(len(c.text) for c in chunks) / len(chunks)),
                        "indexado_s": round(build_s, 2),
                    },
                )
            )
    return pd.DataFrame(rows)


def ood_scores(retriever: Retriever, in_domain: list[dict], out_domain: list[dict]) -> pd.DataFrame:
    """Calcula la similitud del mejor chunk para preguntas de la guía y preguntas ajenas.

    Sirve para ver si se podría usar un umbral de similitud para detectar las
    preguntas que la guía no puede responder.
    """
    rows = [
        {
            "grupo": "dentro de dominio",
            "pregunta": q["pregunta"],
            "max_sim": retriever.max_similarity(q["pregunta"]),
        }
        for q in in_domain
    ]
    rows += [
        {
            "grupo": "fuera de dominio",
            "pregunta": q["pregunta"],
            "max_sim": retriever.max_similarity(q["pregunta"]),
        }
        for q in out_domain
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Evaluación extremo a extremo
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Frases que indican que el asistente reconoce que la guía no trata el tema.
_REFUSAL_HINTS = [
    "no aparece en la guía",
    "no aparece en la guia",
    "la guía no",
    "la guia no",
    "no tengo información",
    "no tengo informacion",
    "no dispongo",
    "no puedo ayudarte con",
    "fuera de",
    "solo puedo",
    "no incluye",
    "no menciona",
    "no recoge",
    "no especifica",
    "no indica",
    "no detalla",
    "no figura",
    "no aparece",
    "no puedo darte",
    "no puedo ayudarte",
]


def _check_case(turn_spec: dict, response) -> dict:
    """Hace las comprobaciones de un turno sobre la respuesta (1.0 = cumple, 0.0 = no cumple)."""
    text = response.text or ""
    norm_text = normalize(text)
    tools_called = [tc["name"] for tc in response.tool_calls]
    checks: dict[str, float] = {}

    if "keywords" in turn_spec:
        kws = [normalize(k) for k in turn_spec["keywords"]]
        checks["keywords"] = sum(k in norm_text for k in kws) / len(kws)
    if turn_spec.get("debe_citar"):
        checks["cita"] = float(bool(_CITATION_RE.search(text)) and bool(response.sources))
    if "herramienta" in turn_spec:
        expected = turn_spec["herramienta"]
        checks["herramienta"] = float(expected in tools_called)
    if turn_spec.get("sin_herramienta_tiempo"):
        checks["herramienta"] = float("get_weather" not in tools_called)
    if turn_spec.get("fuera_de_dominio"):
        checks["rechazo_ood"] = float(any(normalize(h) in norm_text for h in _REFUSAL_HINTS))
    if "no_contiene" in turn_spec:
        checks["sin_fuga"] = float(not any(normalize(t) in norm_text for t in turn_spec["no_contiene"]))
    checks["citas_validas"] = float(not response.invalid_citations)
    if turn_spec.get("error_controlado"):
        weather_calls = [tc for tc in response.tool_calls if tc["name"] == "get_weather"]
        failed = any(not tc.get("ok", True) for tc in weather_calls)
        checks["error_controlado"] = float(failed and len(text) > 0)
    return checks


def run_e2e(make_assistant, cases: list[dict], pause_s: float = 0.0, judge=None) -> pd.DataFrame:
    """Ejecuta los casos de prueba con el asistente real.

    Cada caso tiene una lista de "turnos" y cada turno una "pregunta" y las
    comprobaciones que se esperan. Para cada caso se crea un asistente nuevo,
    así la memoria de un caso no se mezcla con la de otro.

    Args:
        make_assistant: función que devuelve un asistente nuevo.
        cases: casos del fichero eval/e2e_cases.json.
        pause_s: pausa entre turnos (para no pasarse del límite de la API gratuita).
        judge: función opcional (pregunta, contexto, respuesta) -> dict con la
            nota del LLM juez.

    Returns:
        DataFrame con una fila por turno.
    """
    rows = []
    for case in cases:
        assistant = make_assistant()
        for t_idx, turn in enumerate(case["turnos"], start=1):
            try:
                response = assistant.chat(turn["pregunta"])
                error = ""
            except Exception as exc:  # si un caso falla, lo apunto y sigo con los demás
                response, error = None, f"{type(exc).__name__}: {exc}"
            row = {
                "caso": case["id"],
                "categoria": case["categoria"],
                "turno": t_idx,
                "pregunta": turn["pregunta"],
                "error": error,
            }
            if response is not None:
                row.update(
                    {
                        "respuesta": response.text,
                        "herramientas": ", ".join(tc["name"] for tc in response.tool_calls),
                        "n_fuentes": len(response.sources),
                        "latencia_s": response.latency_s,
                        "tokens_entrada": response.usage.get("prompt_tokens", 0),
                        "tokens_salida": response.usage.get("completion_tokens", 0),
                        "llamadas_llm": response.llm_calls,
                        "consulta_busqueda": response.search_query,
                    }
                )
                row.update(_check_case(turn, response))
                if judge is not None and turn.get("juez", True) and response.context:
                    row.update(judge(turn["pregunta"], response.context, response.text))
            rows.append(row)
            if pause_s:
                time.sleep(pause_s)
    return pd.DataFrame(rows)


def summarize_e2e(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la media de cada comprobación por categoría y en total."""
    metric_cols = [
        c
        for c in (
            "keywords",
            "cita",
            "citas_validas",
            "herramienta",
            "rechazo_ood",
            "error_controlado",
            "sin_fuga",
            "fidelidad",
            "precision_citas",
        )
        if c in df.columns
    ]
    by_cat = df.groupby("categoria")[metric_cols + ["latencia_s"]].mean().round(3)
    total = df[metric_cols + ["latencia_s"]].mean().round(3).rename("GLOBAL")
    return pd.concat([by_cat, total.to_frame().T])


JUDGE_PROMPT = """Eres un evaluador estricto de asistentes con RAG.
Te doy una PREGUNTA, el CONTEXTO recuperado de una guía turística y la RESPUESTA del asistente.
Las fuentes del contexto están numeradas [n]; los resultados de herramientas aparecen como [get_weather].
Puntúa de 1 a 5:
- FIDELIDAD: 5 = todo lo que afirma la respuesta está respaldado por el contexto o por las
  herramientas; 1 = inventa información relevante.
- CITAS: 5 = cada cita [n] apunta a una fuente que respalda esa frase; 1 = las citas no corresponden.
  Si la respuesta no tiene citas porque no las necesita (p. ej. solo habla del tiempo), pon 5.
Devuelve SOLO un JSON: {"fidelidad": <1-5>, "citas": <1-5>, "motivo": "<una frase>"}"""


def make_llm_judge(llm, params=None):
    """Crea un LLM juez que puntúa de 1 a 5 la fidelidad y las citas de una respuesta.

    Es mejor que el juez sea un modelo distinto al evaluado, porque un modelo
    tiende a puntuar mejor sus propias respuestas.

    Args:
        llm: el LLMClient.
        params: parámetros del modelo juez (por defecto, los del chat).

    Returns:
        Función (pregunta, contexto, respuesta) -> {"fidelidad", "precision_citas", "motivo_juez"}.
    """

    def judge(question: str, context: str, answer: str) -> dict:
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}",
            },
        ]
        try:
            response = llm.complete(messages, params=params, purpose="judge")
            raw = response.choices[0].message.content or ""
            # El modelo a veces añade texto alrededor del JSON: me quedo solo con lo que va entre llaves.
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            return {
                "fidelidad": float(data.get("fidelidad")),
                "precision_citas": float(data.get("citas", "nan")),
                "motivo_juez": data.get("motivo", ""),
            }
        except Exception as exc:  # si el juez falla, no quiero que se pare toda la evaluación
            return {"fidelidad": None, "precision_citas": None, "motivo_juez": f"juez no disponible: {exc}"}

    return judge
