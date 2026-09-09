"""Persistent session storage with Redis backend + in-memory fallback.

Key format:  gpn:session:{session_id}  →  Redis List of JSON-encoded messages
TTL:         24 hours, reset on every read/append (sliding window)
Fallback:    plain dict if Redis is unreachable or REDIS_URL not set
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_SESSION_TTL = 86400  # 24 hours
_KEY_PREFIX = "gpn:session:"


class SessionStore:
    """Chat history store backed by Redis with automatic in-memory fallback."""

    def __init__(self, redis_url: str | None = None):
        self._redis = None
        self._fallback: dict[str, list[dict]] = {}
        self._fallback_warned = False

        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(
                    redis_url, decode_responses=True, socket_timeout=3,
                )
                self._redis.ping()
                logger.info("Redis подключён: %s", redis_url)
            except Exception as e:
                logger.warning("Redis недоступен (%s) — используется in-memory fallback", e)
                self._redis = None

        if not self._redis:
            logger.info("SessionStore: in-memory fallback (REDIS_URL не задан)")

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def _touch(self, session_id: str) -> None:
        """Reset TTL on the session key."""
        if self._redis:
            try:
                self._redis.expire(self._key(session_id), _SESSION_TTL)
            except Exception:
                pass

    def get_history(self, session_id: str) -> list[dict]:
        """Return full message history for a session."""
        if self._redis:
            try:
                raw = self._redis.lrange(self._key(session_id), 0, -1)
                if raw:
                    self._touch(session_id)
                return [json.loads(m) for m in raw]
            except Exception as e:
                self._log_fallback(e)
        return list(self._fallback.get(session_id, []))

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a single message to the session history."""
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        if self._redis:
            try:
                key = self._key(session_id)
                self._redis.rpush(key, msg)
                self._redis.expire(key, _SESSION_TTL)
                return
            except Exception as e:
                self._log_fallback(e)
        # Fallback: in-memory dict
        if session_id not in self._fallback:
            self._fallback[session_id] = []
        self._fallback[session_id].append({"role": role, "content": content})

    def exists(self, session_id: str) -> bool:
        """Check if a session has any history."""
        if self._redis:
            try:
                return self._redis.exists(self._key(session_id)) > 0
            except Exception as e:
                self._log_fallback(e)
        return session_id in self._fallback

    def delete(self, session_id: str) -> None:
        """Delete a session and all its history."""
        if self._redis:
            try:
                self._redis.delete(self._key(session_id))
                return
            except Exception as e:
                self._log_fallback(e)
        self._fallback.pop(session_id, None)

    def _log_fallback(self, error: Exception) -> None:
        if not self._fallback_warned:
            logger.warning("Redis ошибка, переключаюсь на fallback: %s", error)
            self._fallback_warned = True
