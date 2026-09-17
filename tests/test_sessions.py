"""Тесты хранилища сессий: Redis fallback + TTL-эвикция."""

import time

from sessions import SessionStore


def test_fallback_append_get_exists_delete():
    store = SessionStore(None)
    sid = "s-test-1"

    assert not store.exists(sid)
    store.append_message(sid, "user", "привет")
    store.append_message(sid, "assistant", "здравствуй")

    history = store.get_history(sid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "привет"}
    assert store.exists(sid)

    store.delete(sid)
    assert not store.exists(sid)
    assert store.get_history(sid) == []


def test_fallback_get_returns_copy():
    store = SessionStore(None)
    sid = "s-test-2"
    store.append_message(sid, "user", "x")

    history = store.get_history(sid)
    history.append({"role": "user", "content": "не должен попасть в стор"})

    assert store.get_history(sid) == [{"role": "user", "content": "x"}]


def test_fallback_ttl_eviction():
    store = SessionStore(None)
    sid = "s-test-3"
    store.append_message(sid, "user", "старая сессия")

    # «Старим» метку активности сильнее TTL и запускаем эвикцию
    # (сбрасываем троттлинг, чтобы свип точно выполнился)
    store._fallback_ts[sid] = time.time() - 86400 - 1000
    store._last_sweep = 0.0
    store._fallback_evict()

    assert not store.exists(sid)
    assert store.get_history(sid) == []


def test_fallback_ttl_sliding_window_keeps_active():
    store = SessionStore(None)
    sid = "s-test-4"
    store.append_message(sid, "user", "активная сессия")

    # Активная сессия проживает дольше 24 часов за счёт обновления метки
    store._fallback_ts[sid] = time.time()
    store._last_sweep = 0.0
    store._fallback_evict()

    assert store.exists(sid)


def test_two_sessions_do_not_collide():
    store = SessionStore(None)
    store.append_message("s-a", "user", "A")
    store.append_message("s-b", "user", "B")

    assert store.get_history("s-a")[0]["content"] == "A"
    assert store.get_history("s-b")[0]["content"] == "B"


def test_redis_unreachable_falls_back(tmp_path, monkeypatch):
    """REDIS_URL указан, но Redis не отвечает — работаем через in-memory dict."""
    import redis as _redis

    def fake_from_url(*a, **k):
        class _Client:
            def ping(self):
                raise OSError("no redis")
        return _Client()

    monkeypatch.setattr(_redis.Redis, "from_url", staticmethod(fake_from_url))
    store = SessionStore("redis://127.0.0.1:1/0")
    store.append_message("sid", "user", "ok")
    assert store.get_history("sid") == [{"role": "user", "content": "ok"}]