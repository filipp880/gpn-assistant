async function parseBody(resp) {
  const text = await resp.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function request(path, { method = "GET", headers = {}, body } = {}) {
  const opts = { method, headers };
  if (body !== undefined) {
    if (body instanceof FormData) {
      opts.body = body;
    } else {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
  }
  let resp;
  try {
    resp = await fetch(path, opts);
  } catch {
    throw new Error("Сеть недоступна — проверьте, что сервис запущен");
  }
  const data = await parseBody(resp);
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `HTTP ${resp.status}`;
    throw new Error(typeof detail === "string" ? detail : detail.map((d) => d.msg || "ошибка").join("; "));
  }
  return data;
}

export const api = {
  chat(query, sessionId) {
    const body = sessionId ? { query, session_id: sessionId } : { query };
    return request("/chat", { method: "POST", body });
  },
  health: () => request("/health"),
  history: (sessionId) => request(`/history/${encodeURIComponent(sessionId)}`),
  clearHistory: (sessionId) => request(`/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
  getDictionary: () => request("/dictionary"),
  addTerm: (key, value) => request("/dictionary", { method: "POST", body: { key, value } }),
  deleteTerm: (key) => request(`/dictionary/${encodeURIComponent(key)}`, { method: "DELETE" }),
  upload(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/upload", { method: "POST", body: form });
  },
};