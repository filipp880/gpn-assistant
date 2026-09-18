import { el } from "../dom.js";
import { get, set, subscribe } from "../store.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { addMessage, setBusy } from "./chat.js";

function renderError(err) {
  return {
    role: "assistant",
    error: true,
    content: err ? err.message : "Неизвестная ошибка",
  };
}

export async function sendMessage(rawQuery) {
  const query = String(rawQuery || "").trim();
  if (!query || get("busy")) {
    return;
  }
  const sessionId = get("session_id");
  addMessage({ role: "user", content: query });
  setBusy(true);
  try {
    const resp = await api.chat(query, sessionId);
    set("session_id", resp.session_id || sessionId);
    addMessage({
      role: "assistant",
      content: resp.answer || "",
      sources: resp.sources || [],
      resolved_terms: resp.resolved_terms || [],
      latency: resp.latency_ms,
    });
  } catch (err) {
    addMessage(renderError(err));
  } finally {
    setBusy(false);
  }
}

export function initComposer(container, { onOpenSettings, onOpenDictionary, onOpenTools }) {
  const input = el("textarea", {
    class: "composer-input",
    rows: 1,
    placeholder: "Задайте вопрос…",
    "aria-label": "Сообщение",
  });

  const sendBtn = el("button", {
    class: "send-btn",
    type: "button",
    "aria-label": "Отправить",
    onclick: () => {
      const q = input.value;
      input.value = "";
      autosize();
      sendMessage(q).then(() => input.focus());
    },
  }, icon("send", 18));

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });

  const toolbar = el("div", { class: "composer-toolbar" },
    el("button", { class: "tool-btn", type: "button", title: "Добавить/отредактировать термины словаря", onclick: onOpenDictionary },
      icon("book", 15), "Термины"),
    el("button", { class: "tool-btn", type: "button", title: "Загрузить документ в базу знаний", onclick: onOpenTools },
      icon("upload", 15), "Инструменты"),
    el("button", { class: "tool-btn", type: "button", title: "Настройки интерфейса и статус системы", onclick: onOpenSettings },
      icon("gear", 15), "Настройки"),
  );

  const box = el("div", { class: "composer-box" }, input, sendBtn);

  function enable(flag) {
    sendBtn.disabled = !flag;
  }

  subscribe("busy", (busy) => enable(!busy));

  const composer = el("div", { class: "composer" }, toolbar, box,
    el("div", { class: "composer-hint" },
      "Ассистент может ошибаться. Проверяйте ответы по источникам."));

  container.replaceChildren(composer);

  return { input, enable };
}

export function fillComposer(text) {
  const input = document.querySelector(".composer-input");
  if (input) {
    input.value = text;
    input.dispatchEvent(new Event("input"));
    input.focus();
  }
}