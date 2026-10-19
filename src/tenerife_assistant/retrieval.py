"""Embeddings, vector store (FAISS) y búsqueda híbrida.

- SentenceTransformerEmbedder: embeddings locales y gratis con un modelo
  multilingüe (por defecto intfloat/multilingual-e5-small).
- APIEmbedder: embeddings por API (OpenAI o Gemini), como alternativa.
- DenseIndex: índice FAISS con similitud coseno.
- BM25Index: búsqueda por palabras, útil para nombres propios
  ("Rugantino", "Guachinche El Cubano").
- Retriever: combina los dos índices con Reciprocal Rank Fusion (RRF).

Un "embedder" es cualquier objeto con un atributo `name` y un método
`encode(textos, kind)` que devuelve los vectores normalizados.
"""

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import openai
from rank_bm25 import BM25Okapi

from .ingest import Chunk
from .llm import retry_delay

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def normalize_vectors(vectors) -> np.ndarray:
    """Normaliza cada vector a longitud 1 (así producto escalar = similitud coseno)."""
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SentenceTransformerEmbedder:
    """Embeddings locales con sentence-transformers.

    Los modelos E5 necesitan que los textos empiecen por "query: " o
    "passage: ". Los añado automáticamente si el nombre del modelo tiene "e5".

    Args:
        model_name: modelo de Hugging Face.
        device: "cpu" por defecto para que funcione en cualquier portátil.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = "cpu"):
        # Lo importo aquí porque tarda bastante en cargar y los tests no lo necesitan.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.use_e5_prefix = "e5" in model_name.lower()

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """Convierte textos en vectores. `kind` es "query" (pregunta) o "passage" (fragmento)."""
        if self.use_e5_prefix:
            texts = [f"{kind}: {t}" for t in texts]
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return normalize_vectors(vectors)


class APIEmbedder:
    """Embeddings a través de la API (compatible con OpenAI).

    Sirve para OpenAI (text-embedding-3-small) y para Gemini
    (gemini-embedding-001).
    """

    def __init__(self, client, model_name: str, batch_size: int = 50):
        self.name = model_name
        self.client = client
        self.batch_size = batch_size

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """Convierte textos en vectores llamando a la API por lotes."""
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._create_with_retry(batch)
            vectors.extend(item.embedding for item in response.data)
        return normalize_vectors(np.array(vectors))

    def _create_with_retry(self, batch: list[str], retries: int = 2):
        """Llama a la API; si se supera el límite por minuto (429), espera y reintenta."""
        attempt = 0
        while True:
            try:
                return self.client.embeddings.create(model=self.name, input=batch)
            except openai.RateLimitError as exc:
                # Si es la cuota diaria o ya no quedan intentos, no tiene sentido esperar.
                if attempt == retries or "PerDay" in str(exc):
                    raise
                delay = retry_delay(exc, attempt)
                logger.warning("Límite de la API de embeddings; reintento en %.0f s", delay)
                time.sleep(delay)
                attempt += 1


# ---------------------------------------------------------------------------
# Índices
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Un resultado de búsqueda: el fragmento, su puntuación y su posición."""

    chunk: Chunk
    score: float
    rank: int


class DenseIndex:
    """Vector store FAISS con similitud coseno exacta (IndexFlatIP).

    Con unas decenas de fragmentos la búsqueda exacta tarda microsegundos. Los
    índices aproximados (IVF, HNSW) solo compensan con cientos de miles de vectores.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.index = None

    def build(self, chunks: list[Chunk]) -> "DenseIndex":
        """Calcula los embeddings de los fragmentos y crea el índice."""
        self.chunks = list(chunks)
        vectors = self.embedder.encode([c.embedding_text for c in self.chunks], kind="passage")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Devuelve los k fragmentos más parecidos a la pregunta."""
        if self.index is None:
            raise RuntimeError("El índice no está construido")
        vector = self.embedder.encode([query], kind="query")
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        results = []
        for rank, (score, i) in enumerate(zip(scores[0], ids[0]), start=1):
            if i != -1:  # FAISS devuelve -1 si no hay suficientes resultados
                results.append(SearchResult(self.chunks[i], float(score), rank))
        return results

    def save(self, directory, manifest: dict) -> None:
        """Guarda el índice, los fragmentos y el manifiesto en una carpeta."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.json", "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self.chunks], fh, ensure_ascii=False, indent=1)
        with open(directory / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory, embedder) -> "DenseIndex":
        """Carga un índice guardado con save()."""
        directory = Path(directory)
        obj = cls(embedder)
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.json", encoding="utf-8") as fh:
            obj.chunks = [Chunk.from_dict(d) for d in json.load(fh)]
        return obj


# Palabras muy frecuentes que no ayudan a buscar (se ignoran en BM25).
STOPWORDS = set(
    """a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con
    contra cual cuales cuando de del desde donde dos el ella ellas ellos en entre era es esa
    esas ese eso esos esta estas este esto estos fue ha hay hasta la las le les lo los mas me
    mi mis mucho muy nada ni no nos o os otra otro para pero poco por porque que quien se
    ser si sin sobre son su sus tambien te tiene tu un una uno unos y ya yo vais podeis
    puedo puede hacer hace ver sitio sitios recomiendas recomendais merece pena""".split()
)


def tokenize_es(text: str) -> list[str]:
    """Divide un texto en palabras para BM25: minúsculas, sin tildes y sin stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # quita las tildes
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Índice BM25 (búsqueda por palabras) sobre los mismos fragmentos."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize_es(c.embedding_text) for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Devuelve los k fragmentos con mayor puntuación BM25."""
        scores = self.bm25.get_scores(tokenize_es(query))
        order = np.argsort(-scores)[:k]
        return [SearchResult(self.chunks[i], float(scores[i]), rank) for rank, i in enumerate(order, start=1)]


# ---------------------------------------------------------------------------
# Recuperador
# ---------------------------------------------------------------------------


class Retriever:
    """Buscador que puede usar FAISS, BM25 o los dos (híbrido).

    Args:
        dense: índice FAISS.
        bm25: índice BM25 (hace falta para "bm25" e "hybrid").
        method: "dense", "bm25" o "hybrid".
        rrf_k: constante de RRF (60 es el valor del artículo original).
        candidates: cuántos resultados de cada índice entran en la fusión.
    """

    def __init__(
        self, dense: DenseIndex, bm25=None, method: str = "hybrid", rrf_k: int = 60, candidates: int = 10
    ):
        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Método de recuperación desconocido: {method}")
        if method != "dense" and bm25 is None:
            raise ValueError(f"El método {method} necesita un índice BM25")
        self.dense = dense
        self.bm25 = bm25
        self.method = method
        self.rrf_k = rrf_k
        self.candidates = candidates

    @property
    def chunks(self) -> list[Chunk]:
        """Fragmentos indexados."""
        return self.dense.chunks

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        """Busca la pregunta y devuelve los k mejores fragmentos.

        En modo híbrido cada fragmento suma 1 / (rrf_k + posición) por cada
        índice en el que aparece, y se ordenan por esa suma (RRF).
        """
        if self.method == "dense":
            return self.dense.search(query, k)
        if self.method == "bm25":
            return self.bm25.search(query, k)

        n = max(self.candidates, k)
        fused = {}  # chunk_id -> puntuación RRF
        chunks_by_id = {}
        for results in (self.dense.search(query, n), self.bm25.search(query, n)):
            for r in results:
                cid = r.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + r.rank)
                chunks_by_id[cid] = r.chunk
        ranking = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            SearchResult(chunks_by_id[cid], fused[cid], rank) for rank, cid in enumerate(ranking, start=1)
        ]

    def max_similarity(self, query: str) -> float:
        """Similitud coseno del fragmento más parecido (para estudiar preguntas fuera de dominio)."""
        return self.dense.search(query, 1)[0].score


def build_retriever(chunks, embedder, method="hybrid", cache_dir=None, manifest=None) -> Retriever:
    """Construye el índice FAISS y el BM25, o carga el FAISS de disco si ya existe.

    El manifiesto guarda con qué se construyó el índice (hash del PDF, modelo,
    estrategia de chunking...). Si en cache_dir hay un índice con el mismo
    manifiesto, se reutiliza y no se recalculan los embeddings.

    Args:
        chunks: fragmentos a indexar.
        embedder: modelo de embeddings.
        method: método de búsqueda ("dense", "bm25" o "hybrid").
        cache_dir: carpeta donde guardar el índice (None = no guardar).
        manifest: datos que identifican el índice.

    Returns:
        Un Retriever listo para buscar.
    """
    manifest = dict(manifest or {})
    manifest["embedding_model"] = embedder.name
    manifest["n_chunks"] = len(chunks)

    dense = None
    if cache_dir is not None:
        manifest_path = Path(cache_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            # Estos dos campos cambian en cada construcción, así que no se comparan.
            saved.pop("built_at", None)
            saved.pop("build_seconds", None)
            if saved == manifest:
                dense = DenseIndex.load(cache_dir, embedder)
                logger.info("Índice cargado de caché (%s)", cache_dir)

    if dense is None:
        start = time.perf_counter()
        dense = DenseIndex(embedder).build(chunks)
        elapsed = time.perf_counter() - start
        logger.info("Índice construido en %.2f s", elapsed)
        if cache_dir is not None:
            saved_manifest = {
                **manifest,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "build_seconds": elapsed,
            }
            dense.save(cache_dir, saved_manifest)

    bm25 = BM25Index(dense.chunks) if method != "dense" else None
    return Retriever(dense, bm25, method=method)
