import { el } from "./dom.js";
import { icon } from "./icons.js";

export function openModal({ title, body, footer, onClose } = {}) {
  const overlay = el("div", { class: "modal-overlay", role: "dialog", "aria-modal": "true" });
  const head = el("div", { class: "modal-head" },
    el("h3", {}, title),
    el("button", {
      class: "icon-btn",
      "aria-label": "Закрыть",
      onclick: close,
    }, icon("close")),
  );
  const modal = el("div", { class: "modal" }, head, el("div", { class: "modal-body" }, body));
  if (footer && footer.length) {
    modal.append(el("div", { class: "modal-foot" }, footer));
  }
  overlay.append(modal);

  function close() {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
    if (onClose) {
      onClose();
    }
  }

  function onKey(e) {
    if (e.key === "Escape") {
      close();
    }
  }

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      close();
    }
  });
  document.addEventListener("keydown", onKey);
  document.body.append(overlay);
  return { root: overlay, close, body: modal.querySelector(".modal-body") };
}

export function toast(message, kind = "success") {
  const node = el("div", { class: "toast toast-" + kind, role: "status" }, message);
  document.body.append(node);
  setTimeout(() => node.classList.add("toast-in"), 10);
  setTimeout(() => {
    node.classList.remove("toast-in");
    setTimeout(() => node.remove(), 200);
  }, 2600);
}

export function field(label, input) {
  return el("div", { class: "form-group" }, el("label", {}, label), input);
}