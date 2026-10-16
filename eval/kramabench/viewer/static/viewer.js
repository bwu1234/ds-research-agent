// Sorting and filtering for the viewer's tables. Reads text and data-*
// attributes only; never writes HTML, so ledger text cannot become markup.
"use strict";

function cellKey(td) {
  const v = td.dataset.v !== undefined ? td.dataset.v : td.textContent.trim();
  const n = Number(v);
  return v !== "" && !Number.isNaN(n) ? n : v.toLowerCase();
}

function sortBy(table, th) {
  const col = Array.from(th.parentNode.children).indexOf(th);
  const dir = th.dataset.dir === "asc" ? "desc" : "asc";
  table.querySelectorAll("th").forEach((h) => delete h.dataset.dir);
  th.dataset.dir = dir;
  const body = table.tBodies[0];
  const rows = Array.from(body.rows);
  const sign = dir === "asc" ? 1 : -1;
  rows.sort((a, b) => {
    const x = cellKey(a.cells[col]);
    const y = cellKey(b.cells[col]);
    if (x === y) return 0;
    if (x === "") return 1; // blanks last either way
    if (y === "") return -1;
    if (typeof x !== typeof y) return typeof x === "number" ? -sign : sign;
    return x < y ? -sign : sign;
  });
  rows.forEach((r) => body.appendChild(r));
}

function applyFilters(id) {
  const table = document.getElementById(id);
  if (!table) return;
  const text = document.querySelector(`[data-filter-for="${id}"]`);
  const outcome = document.querySelector(`[data-outcome-for="${id}"]`);
  const q = text ? text.value.trim().toLowerCase() : "";
  const want = outcome ? outcome.value : "";
  let shown = 0;
  const rows = Array.from(table.tBodies[0].rows);
  rows.forEach((r) => {
    const ok =
      (!q || r.textContent.toLowerCase().includes(q)) &&
      (!want || r.dataset.outcome === want || (want === "strict" && r.dataset.outcome === "unverified"));
    r.hidden = !ok;
    if (ok) shown += 1;
  });
  const count = document.querySelector(`[data-count-for="${id}"]`);
  if (count) count.textContent = `${shown} of ${rows.length} rows`;
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("table.sortable").forEach((table) => {
    table.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => sortBy(table, th)));
    applyFilters(table.id);
  });
  document.querySelectorAll("[data-filter-for]").forEach((el) =>
    el.addEventListener("input", () => applyFilters(el.dataset.filterFor)));
  document.querySelectorAll("[data-outcome-for]").forEach((el) =>
    el.addEventListener("change", () => applyFilters(el.dataset.outcomeFor)));
});
