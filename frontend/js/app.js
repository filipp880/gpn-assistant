import { get, set, subscribe } from "./store.js";
import { api } from "./api.js";
import { initHeader } from "./views/header.js";
import { initChatView, clearMessages, addMessage } from "./views/chat.js";
import { initComposer, fillComposer } from "./views/composer.js";
import { openSettingsModal, openDictionaryModal, openUploadModal, openProfileModal } from "./views/modals.js";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="color-scheme"]');
  if (meta) {
    meta.content = theme;
  }
}

function newChat() {
  const sid = get("session_id");
  clearMessages();
  if (sid) {
    set("session_id", null);
    api.clearHistory(sid).catch(() => {});
  }
  document.querySelector(".composer-input")?.focus();
}

async function restoreHistory() {
  const sid = get("session_id");
  if (!sid) {
    return;
  }
  try {
    const data = await api.history(sid);
    const history = (data && data.history) || [];
    for (const msg of history) {
      if (msg.role === "user") {
        addMessage({ role: "user", content: msg.content });
      } else if (msg.role === "assistant") {
        addMessage({ role: "assistant", content: msg.content });
      }
    }
  } catch {
    set("session_id", null);
  }
}

async function bootstrap() {
  applyTheme(get("theme"));
  subscribe("theme", applyTheme);

  const headerEl = document.getElementById("app-header");
  const chatEl = document.getElementById("chat-view");
  const composerEl = document.getElementById("composer");

  initHeader(headerEl, { onNewChat: newChat, onProfile: openProfileModal });
  initChatView(chatEl, { onSuggestion: fillComposer });
  initComposer(composerEl, {
    onOpenSettings: openSettingsModal,
    onOpenDictionary: openDictionaryModal,
    onOpenTools: openUploadModal,
  });

  await restoreHistory();
}

bootstrap();