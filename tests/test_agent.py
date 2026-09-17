"""Тесты агента: полный пайплайн process_query с моками + контракт /chat."""

import os

os.environ["WARMUP_MODELS"] = "0"  # до импорта app — не даём фоновому прогреву грузить модели

import pytest

import agent
from agent import GpnAgent, run_my_agent_logic


# --- process_query с замоканными LLM и поиском ---

@pytest.fixture()
def mocked_agent(monkeypatch):
    """GpnAgent с мокнутыми decompose/LLM — поиск подменяется в самом тесте."""
    inst = GpnAgent()
    inst.self_check = False

    def fake_decompose(self, user_query, trace=None):
        return ["EBITDA компании"]

    monkeypatch.setattr(GpnAgent, "_decompose_query", fake_decompose)

    def fake_chat(messages, options=None, **kwargs):
        return {"message": {"content": "EBITDA растёт."}, "usage": {"total_tokens": 5}}

    monkeypatch.setattr(inst, "_chat", fake_chat)
    return inst


def test_process_query_mocked_pipeline(monkeypatch, mocked_agent):
    def fake_hybrid(query, top_k=None, use_reranker=True, trace=None):
        return [("текст документа про EBITDA за 2025", "doc1.txt")]

    monkeypatch.setattr(agent, "hybrid_search", fake_hybrid)

    result = mocked_agent.process_query("Какая EBITDA у ГПН-Р?", [])

    assert result["answer"] == "EBITDA растёт."
    assert result["sources"] == ["doc1.txt"]
    assert any(t["original"] == "ГПН-Р" for t in result["resolved_terms"])
    assert "текст документа про EBITDA" in result["raw_context"]


def test_process_query_no_context_answer(monkeypatch, mocked_agent):
    monkeypatch.setattr(agent, "hybrid_search", lambda *a, **k: [])
    result = mocked_agent.process_query("про что нет в базе?", [])
    assert isinstance(result["answer"], str)
    assert result["raw_context"] == ""


def test_run_my_agent_logic_delegates(monkeypatch):
    calls = []

    class FakeAgent:
        def process_query(self, query, history, trace=None):
            calls.append((query, history))
            return {"answer": "ok", "sources": [], "resolved_terms": [], "raw_context": ""}

    monkeypatch.setattr(agent, "get_agent", lambda: FakeAgent())

    res = run_my_agent_logic("вопрос", "sid-1", [{"role": "user", "content": "вопрос"}])
    assert res["answer"] == "ok"
    assert calls[0][0] == "вопрос"


def test_get_agent_is_singleton():
    assert agent.get_agent() is agent.get_agent()


# --- Контракт /chat через TestClient (lifespan без прогрева) ---

def test_chat_endpoint_contract(monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module
    from routes import chat as chat_router

    def fake_run(query, session_id, history, trace=None):
        return {
            "answer": "тест ответ",
            "sources": ["doc.txt"],
            "resolved_terms": [],
            "raw_context": "ctx",
        }

    monkeypatch.setattr(chat_router, "_run_sync_agent", fake_run)

    with TestClient(app_module.app) as client:
        r = client.post("/chat", json={"query": "вопрос"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "тест ответ"
        assert body["sources"] == ["doc.txt"]
        assert body["session_id"]

        sid = body["session_id"]
        h = client.get(f"/history/{sid}")
        assert h.status_code == 200
        messages = h.json()["history"]
        assert [m["role"] for m in messages] == ["user", "assistant"]

        assert client.delete(f"/history/{sid}").status_code == 200
        assert client.get(f"/history/{sid}").status_code == 404


def test_chat_returns_503_when_ollama_down(monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module
    from agent import OllamaUnavailableError
    from routes import chat as chat_router

    def fake_run(query, session_id, history, trace=None):
        raise OllamaUnavailableError("ollama dead")

    monkeypatch.setattr(chat_router, "_run_sync_agent", fake_run)

    with TestClient(app_module.app) as client:
        r = client.post("/chat", json={"query": "вопрос"})
        assert r.status_code == 503
        assert "Ollama" in r.json()["detail"]