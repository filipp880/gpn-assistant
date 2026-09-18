import { el } from "../dom.js";
import { get, set, subscribe } from "../store.js";
import { icon } from "../icons.js";

export function initHeader(container, { onNewChat, onProfile }) {
  function render() {
    const theme = get("theme");
    const user = get("user");

    const themeBtn = el("button", {
      class: "icon-btn",
      title: theme === "dark" ? "Светлая тема" : "Тёмная тема",
      "aria-label": "Переключить тему",
      onclick: () => set("theme", theme === "dark" ? "light" : "dark"),
    }, icon(theme === "dark" ? "sun" : "moon", 18));

    const newChatBtn = el("button", {
      class: "btn btn-ghost",
      title: "Начать новый чат",
      onclick: onNewChat,
    }, icon("plus", 16), "Новый чат");

    const loginBtn = user
      ? el("button", {
        class: "user-chip",
        title: user.name + (user.role ? " · " + user.role : ""),
        onclick: onProfile,
      },
      el("span", { class: "user-avatar" }, user.name.trim().charAt(0).toUpperCase()),
      el("span", { class: "user-name" }, user.name))
      : el("button", {
        class: "tool-btn",
        onclick: onProfile,
        title: "Войти",
      }, icon("user", 15), "Войти");

    container.replaceChildren(
      el("div", { class: "header-brand" },
        el("span", { class: "brand-mark" }, icon("logo", 18)),
        el("span", { class: "brand-text" }, "GPN Assistant")),
      newChatBtn,
      el("div", { class: "header-spacer" }),
      el("div", { class: "header-actions" }, themeBtn, loginBtn),
    );
  }

  subscribe(["theme", "user"], render);
  render();
}