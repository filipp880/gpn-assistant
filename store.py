"""Единый экземпляр хранилища сессий (Redis + in-memory fallback)."""

import os

from sessions import SessionStore

store = SessionStore(os.getenv("REDIS_URL"))