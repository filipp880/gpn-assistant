"""Тесты retrieval: sparse-скоринг, fingerprint и кэш индекса (без моделей)."""

import math

import pytest

from retrieval import (
    _docs_fingerprint,
    _load_sparse_cache,
    _save_sparse_cache,
    _sparse_scores,
    BM25_WEIGHT,
)


def _inverted(pairs):
    """Строит инвертированный индекс {token: [(doc_idx, weight)]} из пар."""
    inv = {}
    for doc_idx, weights in pairs:
        for tok, w in weights.items():
            inv.setdefault(tok, []).append((doc_idx, w))
    return inv


def test_sparse_scores_matches_bruteforce():
    """SPLADE-скоринг через общий движок == прямому произведению весов."""
    docs = [
        {"EBITDA": 2.0, "2025": 1.5},
        {"EBITDA": 1.0, "добыча": 3.0},
        {"нефть": 0.5},
    ]
    inverted = _inverted(list(enumerate(docs)))
    query_weights = {"EBITDA": 1.2, "добыча": 0.9}

    scores = _sparse_scores(query_weights, inverted, n_docs=len(docs))

    assert len(scores) == 3
    assert scores[0] == pytest.approx(2.0 * 1.2)
    assert scores[1] == pytest.approx(1.0 * 1.2 + 3.0 * 0.9)
    assert scores[2] == 0.0


def test_sparse_scores_weighted_sum_ranking():
    """Взвешенная сумма по пересечению токенов: больший суммарный вклад — выше."""
    inverted = {
        "нефть": [(0, 2.0), (1, 1.0)],
        "ГПН": [(0, 0.5)],
    }
    scores = _sparse_scores({"нефть": 1.0, "ГПН": 1.0}, inverted, n_docs=2)
    assert scores[0] == pytest.approx(2.5)
    assert scores[1] == pytest.approx(1.0)
    assert scores.index(max(scores)) == 0


def test_docs_fingerprint_stable_and_sensitive():
    a = ["один документ", "второй документ"]
    b = ["один документ", "второй документ"]
    c = ["один документ", "ВТОРОЙ документ"]

    assert _docs_fingerprint(a) == _docs_fingerprint(b)
    assert _docs_fingerprint(a) != _docs_fingerprint(c)


def test_sparse_cache_roundtrip(tmp_path, monkeypatch):
    from retrieval import SPARSE_CACHE_FILE
    cache_file = tmp_path / "sparse_index.json"
    monkeypatch.setattr("retrieval.SPARSE_CACHE_FILE", str(cache_file))

    docs = [
        {"EBITDA": 1.5, "ГПНР": 0.8},
        {"добыча": 2.0},
    ]
    ids = ["doc1", "doc2"]
    fp = _docs_fingerprint(["текст1", "текст2"])

    _save_sparse_cache(fp, ids, docs)

    loaded = _load_sparse_cache(fp, ids)
    assert loaded is not None
    assert loaded["EBITDA"] == [(0, 1.5)]
    assert loaded["добыча"] == [(1, 2.0)]


def test_sparse_cache_invalidated_on_corpus_change(tmp_path, monkeypatch):
    from retrieval import SPARSE_CACHE_FILE
    cache_file = tmp_path / "sparse_index.json"
    monkeypatch.setattr("retrieval.SPARSE_CACHE_FILE", str(cache_file))

    docs = [{"EBITDA": 1.5}]
    ids = ["doc1"]
    fp_old = _docs_fingerprint(["старый корпус"])
    fp_new = _docs_fingerprint(["новый корпус"])

    _save_sparse_cache(fp_old, ids, docs)

    assert _load_sparse_cache(fp_new, ids) is None


def test_estimate_top_k_smoke():
    from core import estimate_top_k
    k = estimate_top_k("Какой простой вопрос?")
    assert isinstance(k, int) and k >= 1


def test_bm25_weight_is_positive_constant():
    assert BM25_WEIGHT > 0 and BM25_WEIGHT <= 1.0
    assert math.isfinite(BM25_WEIGHT)