import { el } from "../dom.js";
import { get, set } from "../store.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { openModal, toast, field } from "../ui.js";

let dictionaryCache = null;

export async function openDictionaryModal() {
  const status = el("div", { class: "form-status" });
  const listBox = el("div", { class: "dict-list" }, el("div", { class: "dict-empty" }, "Загрузка словаря…"));

  const search = el("input", {
    class: "field dict-search",
    type: "search",
    placeholder: "Поиск по терминам…",
  });

  const termInput = el("input", { class: "field", type: "text", placeholder: "Например: КРС" });
  const defInput = el("input", {
    class: "field",
    type: "text",
    placeholder: "Например: Капитальный ремонт скважин",
  });

  const addBtn = el("button", { class: "btn btn-primary", type: "button" }, "Добавить");

  function renderDict() {
    const data = dictionaryCache || {};
    const q = search.value.trim().toLowerCase();
    const entries = Object.entries(data).filter(([term, def]) =>
      !q || term.toLowerCase().includes(q) || def.toLowerCase().includes(q));

    if (!entries.length) {
      listBox.replaceChildren(el("div", { class: "dict-empty" },
        q ? "Ничего не найдено" : "Словарь пуст"));
      return;
    }

    const rows = entries.map(([term, def]) =>
      el("div", { class: "dict-item" },
        el("code", { class: "dict-key" }, term),
        el("div", { class: "dict-value" }, def),
        el("button", {
          class: "icon-btn",
          title: "Удалить " + term,
          "aria-label": "Удалить " + term,
          onclick: async () => {
            try {
              const resp = await api.deleteTerm(term);
              dictionaryCache = resp.dictionary;
              renderDict();
              toast("Термин удалён");
            } catch (err) {
              toast(err.message, "error");
            }
          },
        }, icon("trash", 16)),
      ));

    listBox.replaceChildren(el("div", { class: "dict-list" }, rows));
  }

  search.addEventListener("input", renderDict);

  addBtn.addEventListener("click", async () => {
    const term = termInput.value.trim();
    const def = defInput.value.trim();
    if (!term || !def) {
      status.textContent = "Заполните термин и расшифровку";
      status.className = "form-status error";
      return;
    }
    addBtn.disabled = true;
    status.textContent = "Сохранение…";
    status.className = "form-status";
    try {
      const resp = await api.addTerm(term, def);
      dictionaryCache = resp.dictionary;
      termInput.value = "";
      defInput.value = "";
      status.textContent = "Термин добавлен";
      status.className = "form-status success";
      renderDict();
    } catch (err) {
      status.textContent = err.message;
      status.className = "form-status error";
    } finally {
      addBtn.disabled = false;
    }
  });

  const modal = openModal({
    title: "Корпоративный словарь терминов",
    body: el("div", {},
      status,
      el("div", { class: "form-row" }, termInput, defInput),
      el("div", {}, addBtn),
      search,
      listBox),
  });

  try {
    const data = await api.getDictionary();
    dictionaryCache = data && data.dictionary ? data.dictionary : {};
    renderDict();
  } catch (err) {
    listBox.replaceChildren(el("div", { class: "dict-empty error" }, err.message));
  }

  return modal;
}

export function openUploadModal() {
  const status = el("div", { class: "form-status" });
  const fileName = el("div", { class: "form-hint" });

  const fileInput = el("input", { class: "field", type: "file", accept: ".pdf,.txt,.docx,.xlsx,.pptx" });

  const dropZone = el("div", { class: "drop-zone" },
    icon("upload", 22),
    el("div", {}, "Перетащите файл сюда или нажмите для выбора"),
    el("div", { class: "form-hint" }, "PDF, TXT, DOCX, XLSX, PPTX"));

  dropZone.addEventListener("click", () => fileInput.click());

  let dragDepth = 0;
  dropZone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) {
      dropZone.classList.remove("dragover");
    }
  });
  dropZone.addEventListener("dragover", (e) => e.preventDefault());
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      selectFile(fileInput.files[0]);
    }
  });

  function selectFile(file) {
    fileName.textContent = "Выбран файл: " + file.name;
    status.textContent = "";
    status.className = "form-status";
  }

  const uploadBtn = el("button", { class: "btn btn-primary", type: "button" }, "Загрузить и переиндексировать");

  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      status.textContent = "Выберите файл";
      status.className = "form-status error";
      return;
    }
    uploadBtn.disabled = true;
    status.textContent = "Загрузка и переиндексация… это может занять время";
    status.className = "form-status";
    try {
      const resp = await api.upload(file);
      status.textContent = resp.message || "Готово";
      status.className = "form-status success";
      toast("Документ добавлен в базу знаний");
    } catch (err) {
      status.textContent = err.message;
      status.className = "form-status error";
    } finally {
      uploadBtn.disabled = false;
    }
  });

  const modal = openModal({
    title: "Инструменты — база знаний",
    body: el("div", {},
      dropZone,
      fileInput,
      fileName,
      uploadBtn,
      status,
      el("div", { class: "form-hint" },
        "Файл сохраняется в data/ и индекс пересобирается автоматически (POST /upload).")),
  });

  return modal;
}

export function openSettingsModal() {
  const themeState = get("theme");

  const themeLight = el("input", { type: "radio", name: "theme", id: "theme-light", value: "light" });
  const themeDark = el("input", { type: "radio", name: "theme", id: "theme-dark", value: "dark" });
  themeLight.checked = themeState !== "dark";
  themeDark.checked = themeState === "dark";

  for (const radio of [themeLight, themeDark]) {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        set("theme", radio.value);
      }
    });
  }

  const sourcesToggle = el("input", { type: "checkbox", id: "show-sources" });
  sourcesToggle.checked = get("show_sources");
  sourcesToggle.addEventListener("change", () => set("show_sources", sourcesToggle.checked));

  const statusId = "sys-" + Math.random().toString(36).slice(2);
  const sysBox = el("div", { id: statusId, class: "info-list" });
  sysBox.append(el("div", { class: "info-row" }, el("span", {}, "Загрузка…"), el("span", {}, "")));

  const refreshBtn = el("button", {
    class: "icon-btn",
    title: "Обновить статус",
    onclick: loadHealth,
  }, icon("refresh", 15));

  async function loadHealth() {
    sysBox.replaceChildren(el("div", { class: "info-row", style: "justify-content:flex-start;column-gap:8px" },
      el("span", { class: "spinner" }), "Проверка сервиса…"));
    try {
      const h = await api.health();
      const badge = (value) => {
        if (value === true || value === "up" || value === "ok" || value === "ready") {
          return el("span", { class: "badge badge-ok" }, "OK");
        }
        return el("span",
          { class: value === "down" || value === "not_ready" ? "badge badge-bad" : "badge badge-warn" },
          String(value));
      };
      const rows = [
        ["Статус сервиса", badge(h.status === "ok")],
        ["Ollama", badge(h.ollama_status)],
        ["База знаний", badge(h.db_status)],
        ["Модели", badge(h.models_ready)],
        ["Сообщение", h.message || "—"],
      ];
      sysBox.replaceChildren(
        ...rows.map(([key, value]) =>
          el("div", { class: "info-row" }, el("span", { class: "info-key" }, key), el("span", {}, value))));
    } catch (err) {
      sysBox.replaceChildren(el("div", { class: "form-status error" }, err.message));
    }
  }

  loadHealth();

  const modal = openModal({
    title: "Настройки",
    body: el("div", {},
      el("div", { class: "form-group", style: "display:grid;gap:8px" },
        el("label", {}, "Тема"),
        el("label", { class: "check-row" }, themeLight, "Светлая"),
        el("label", { class: "check-row" }, themeDark, "Тёмная")),
      el("label", { class: "check-row" }, sourcesToggle, "Показывать источники и время ответа"),
      el("div", {},
        el("div", { style: "display:flex;align-items:center;gap:8px" },
          el("label", { class: "info-key", style: "flex:1" }, "Статус системы"),
          refreshBtn),
        sysBox)),
  });

  return modal;
}

export function openProfileModal() {
  const user = get("user");
  const status = el("div", { class: "form-status" });

  const nameInput = el("input", { class: "field", type: "text", placeholder: "Имя и фамилия", value: user ? user.name : "" });
  const roleInput = el("select", { class: "field" },
    el("option", { value: "Сотрудник" }, "Сотрудник"),
    el("option", { value: "Аналитик" }, "Аналитик"),
    el("option", { value: "Менеджер" }, "Менеджер"),
    el("option", { value: "Гость" }, "Гость"));
  roleInput.value = user ? (user.role || "Сотрудник") : "Сотрудник";

  const saveBtn = el("button", { class: "btn btn-primary", type: "button" }, "Сохранить");

  saveBtn.addEventListener("click", () => {
    const name = nameInput.value.trim();
    if (!name) {
      status.textContent = "Введите имя";
      status.className = "form-status error";
      return;
    }
    set("user", { name, role: roleInput.value });
    status.textContent = "Профиль сохранён (демо-авторизация без сервера)";
    status.className = "form-status success";
  });

  const logoutBtn = el("button", { class: "btn btn-danger", type: "button" }, "Выйти");
  logoutBtn.addEventListener("click", () => {
    set("user", null);
    status.textContent = "Вы вышли";
    status.className = "form-status success";
    nameInput.value = "";
  });

  const body = field("Имя", nameInput);
  body.append(field("Роль", roleInput), status);

  const footer = [];
  if (user) {
    footer.push(logoutBtn);
  }
  footer.push(saveBtn);

  return openModal({
    title: user ? "Профиль пользователя" : "Вход",
    body,
    footer,
  });
}