import pytest

from tenerife_assistant.evaluation import evaluate_retrieval, summarize_retrieval
from tenerife_assistant.retrieval import BM25Index, DenseIndex, Retriever, build_retriever, tokenize_es

from .conftest import HashEmbedder


def test_tokenizer_removes_accents_and_stopwords():
    assert tokenize_es("¿Dónde está el Drago Milenario?") == ["drago", "milenario"]


def test_bm25_finds_proper_nouns(chunks):
    results = BM25Index(chunks).search("pizzería Rugantino", k=3)
    assert "Rugantino" in results[0].chunk.text


def test_hybrid_returns_k_unique_results(retriever):
    results = retriever.search("playa de arena negra para ver el atardecer en Anaga", k=4)
    assert len(results) == 4
    assert len({r.chunk.chunk_id for r in results}) == 4
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert any("Benijo" in r.chunk.text for r in results)


def test_invalid_method_raises(chunks):
    dense = DenseIndex(HashEmbedder()).build(chunks)
    with pytest.raises(ValueError):
        Retriever(dense, None, method="hybrid")


def test_index_is_cached_on_disk(chunks, tmp_path):
    embedder = HashEmbedder()
    manifest = {"pdf_sha256": "abc", "chunk_strategy": "structural"}
    build_retriever(chunks, embedder, "hybrid", tmp_path, manifest)
    calls_after_build = embedder.calls
    retriever = build_retriever(chunks, embedder, "hybrid", tmp_path, manifest)
    assert embedder.calls == calls_after_build  # no se recalculan los embeddings
    assert len(retriever.chunks) == len(chunks)
    # Si cambia el manifiesto (otro PDF o estrategia), se reconstruye.
    build_retriever(chunks, embedder, "hybrid", tmp_path, {**manifest, "pdf_sha256": "otro"})
    assert embedder.calls > calls_after_build


def test_retrieval_metrics(retriever):
    questions = [
        {"id": "a", "pregunta": "pizzería Rugantino Garachico", "evidencias": ["Rugantino"]},
        {"id": "b", "pregunta": "zzz", "evidencias": ["texto que no existe"]},
    ]
    df = evaluate_retrieval(retriever, questions, k_values=(1, 5))
    summary = summarize_retrieval(df)
    assert df.loc[0, "hit@5"] == 1.0 and df.loc[1, "hit@5"] == 0.0
    assert 0 < summary["MRR"] <= 0.5
