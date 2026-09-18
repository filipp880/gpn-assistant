import { el, esc } from "../dom.js";
import { get, set, subscribe } from "../store.js";
import { icon } from "../icons.js";

const SUGGESTIONS = [
  "Что такое ГРП и как ПНГ влияет на экологию?",
  "Какие показатели учитываются в отчёте по ГПНР?",
  "Что такое RAG и зачем он нужен?",
  "Какая EBITDA у ГПНР за 2025 год?",
];

function userMark() {
  const user = get("user");
  return user && user.name ? esc(user.name.trim().charAt(0).toUpperCase() || "П") : "Вы";
}

function assistantMessage(msg) {
  const showSources = get("show_sources");
  const body = document.createElement("div");
  body.className = "msg-body";

  const content = el("div", { class: "msg-content" }, msg.content);
  body.append(content);

  const metaChildren = [];
  if (msg.latency != null) {
    metaChildren.push(el("span", {}, "⏱ " + Math.round(msg.latency) + " мс"));
  }
  if (showSources && msg.sources && msg.sources.length) {
    for (const src of msg.sources) {
      metaChildren.push(el("span", { class: "source-chip", title: src },
        icon("file", 13), el("span", {}, esc(src))));
    }
  }
  if (metaChildren.length) {
    body.append(el("div", { class: "msg-meta" }, metaChildren));
  }

  if (msg.resolved_terms && msg.resolved_terms.length) {
    const rows = msg.resolved_terms.map((t) =>
      el("div", {},
        el("strong", {}, esc(t.original)),
        " → " + esc(t.canonical),
        el("span", { style: "color:var(--text-tertiary)" }, ` (${Math.round(t.score)})`)));
    body.append(el("details", { class: "msg-details" },
      el("summary", {}, "Термины из корпоративного словаря"),
      el("div", { class: "msg-details-list" }, rows),
    ));
  }

  return el("div", { class: "msg msg-assistant" },
    el("div", { class: "msg-avatar", title: "GPN Assistant" }, icon("logo", 16)),
    body);
}

function userMessage(msg) {
  return el("div", { class: "msg msg-user" },
    el("div", { class: "msg-body" }, el("div", { class: "msg-content" }, esc(msg.content))),
    el("div", { class: "msg-avatar", title: "Вы" }, userMark()));
}

function errorMessage(msg) {
  return el("div", { class: "msg msg-assistant msg-error" },
    el("div", { class: "msg-avatar" }, icon("info", 16)),
    el("div", { class: "msg-body" }, el("div", { class: "msg-content" }, esc(msg.content))));
}

function emptyState(onSuggestion) {
  return el("div", { class: "chat-empty", "data-testid": "chat-empty" },
    el("div", { class: "msg-avatar", style: "width:44px;height:44px;border-radius:14px;font-size:22px" }, icon("logo", 22)),
    el("div", { class: "chat-empty-title" }, "Корпоративный AI-ассистент"),
    el("div", { class: "chat-empty-sub" },
      "Понимает язык компании: распознаёт внутренние аббревиатуры, исправляет опечатки и отвечает по базе знаний."),
    el("div", { class: "suggestions" },
      SUGGESTIONS.map((text) =>
        el("button", { class: "suggestion-chip", onclick: () => onSuggestion(text) }, esc(text)))),
  );
}

function typingIndicator() {
  return el("div", { class: "msg msg-assistant" },
    el("div", { class: "msg-avatar" }, icon("logo", 16)),
    el("div", { class: "msg-body" },
      el("div", { class: "typing-indicator", "aria-label": "Ассистент пишет", role: "status" },
        el("span", { class: "typing-dot" }), el("span", { class: "typing-dot" }), el("span", { class: "typing-dot" }))));
}

export function initChatView(container, { onSuggestion }) {
  const scroll = document.getElementById("chat-scroll");

  function render() {
    const messages = get("messages");
    const busy = get("busy");
    const children = [];

    if (!messages.length) {
      children.push(emptyState(onSuggestion));
    }
    for (const msg of messages) {
      if (msg.role === "user") {
        children.push(userMessage(msg));
      } else if (msg.error) {
        children.push(errorMessage(msg));
      } else {
        children.push(assistantMessage(msg));
      }
    }
    if (busy) {
      children.push(typingIndicator());
    }

    container.replaceChildren(...children);
    if (scroll) {
      scroll.scrollTop = scroll.scrollHeight;
    }
  }

  subscribe(["messages", "busy", "show_sources", "user"], render);
  render();
  return { render };
}

export function addMessage(msg) {
  set("messages", [...get("messages"), msg]);
}

export function clearMessages() {
  set("messages", []);
}

export function setBusy(value) {
  set("busy", value);
}