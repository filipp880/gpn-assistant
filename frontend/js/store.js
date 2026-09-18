const state = Object.create(null);
const listeners = new Map();

const PERSIST_KEYS = ["theme", "user", "session_id"];

function restore(key, fallback) {
  try {
    const raw = localStorage.getItem("gpn." + key);
    return raw != null ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function persist(key, value) {
  try {
    if (value == null) {
      localStorage.removeItem("gpn." + key);
    } else {
      localStorage.setItem("gpn." + key, JSON.stringify(value));
    }
  } catch {
    /* localStorage недоступен — молча продолжаем в памяти */
  }
}

state.theme = restore("theme", "light");
state.user = restore("user", null);
state.session_id = restore("session_id", null);
state.messages = [];
state.busy = false;
state.show_sources = true;

export function get(key) {
  return state[key];
}

export function set(key, value) {
  state[key] = value;
  if (PERSIST_KEYS.includes(key)) {
    persist(key, value);
  }
  for (const [fn, keys] of listeners) {
    if (keys.includes(key)) {
      fn(value, key);
    }
  }
}

export function subscribe(keys, fn) {
  const list = Array.isArray(keys) ? keys : [keys];
  listeners.set(fn, list);
  return () => listeners.delete(fn);
}