const domain = window.document.location.href.slice(
  0,
  window.document.location.href.indexOf("main")
);

export function openWindow(path: string) {
  window.open(domain + path, "_blank");
}
