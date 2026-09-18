const PATHS = {
  logo: "M13.5 2 6 13h5l-1 9 8-12h-5l1-8Z",
  plus: "M12 5v14M5 12h14",
  send: "m22 2-7 20-4-9-9-4Z",
  gear: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.35-2.72a7.6 7.6 0 0 0 0-2.56l2-1.55-2-3.46-2.35.95a7.5 7.5 0 0 0-2.22-1.28L14.4 2H8.6l-.38 2.38a7.5 7.5 0 0 0-2.22 1.28l-2.35-.95-2 3.46 2 1.55a7.6 7.6 0 0 0 0 2.56l-2 1.55 2 3.46 2.35-.95a7.5 7.5 0 0 0 2.22 1.28L8.6 22h5.8l.38-2.38a7.5 7.5 0 0 0 2.22-1.28l2.35.95 2-3.46-2-1.55Z",
  book: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20V4a2 2 0 0 0-2-2H6.5A2.5 2.5 0 0 0 4 4.5v15ZM4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  user: "M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0-15v2m0 16v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M2 12h2m16 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42",
  moon: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z",
  trash: "M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6",
  close: "M18 6 6 18M6 6l12 12",
  search: "m21 21-4.35-4.35M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z",
  refresh: "M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6",
  file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8ZM14 2v6h6",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Zm0-12v6m0-10h.01",
};

const TEMPLATE =
  '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="__PATH__"/></svg>';

export function icon(name, size = 20) {
  const path = PATHS[name];
  if (!path) {
    return "";
  }
  return TEMPLATE.replace("__PATH__", path).replace(`width="20"`, `width="${size}"`).replace(`height="20"`, `height="${size}"`);
}