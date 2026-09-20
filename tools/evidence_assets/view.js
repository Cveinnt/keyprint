"use strict";
const evidence = window.KEYPRINT_EVIDENCE;
const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const scoreText = value => Number(value).toFixed(3);
const categoryName = value => value.replaceAll("_", " ");
const controls = evidence.controls;
const available = controls.filter(r => !r.error);
const hits = available.filter(r => r.flagged).length;
text("marked-count", `${evidence.cases.filter(c => c.marked.matching_flagged).length} / ${evidence.cases.length}`);
text("ordinary-count", `${evidence.cases.filter(c => c.ordinary.scores.some(s => s >= evidence.cutoff)).length} / ${evidence.cases.length}`);
text("human-count", `${hits} / ${available.length}`);
text("human-scope", evidence.partial ? `Audited snapshot · ${evidence.planned_controls} planned` : `${available.length} available / ${evidence.planned_controls} planned`);
const complete = !evidence.partial && evidence.null_summary?.status === "completed";
text("state", evidence.partial ? `Study running. This snapshot contains ${controls.length} of ${evidence.planned_controls} planned human controls. No final false-positive result.` : complete ? evidence.recovery ? `Completed across two executions. ${evidence.null_summary.null_screen_passed ? "Combined observation screen passed" : "Combined observation screen failed"}; the original attempt remains incomplete. Public release held.` : `Study complete. ${evidence.null_summary.null_screen_passed ? "Frozen corpus screen passed" : "Frozen corpus screen failed"}. Public release remains held.` : `Study incomplete. ${evidence.planned_controls - available.length} of ${evidence.planned_controls} controls are unavailable. No false-positive bound or passing claim.`);
text("bound", complete ? `Human-control flags: ${hits}/${available.length}. One-sided 97.5% IID-only upper bound: ${(100 * evidence.null_summary.iid_only_upper_97_5_percent).toFixed(4)}%. The bound depends on an independence assumption this corpus cannot establish.` : "A false-positive bound will only be shown after all planned controls finish and the final integrity audit passes. These scores are not probabilities of AI authorship.");
text("cutoff", `log(200) ≈ ${scoreText(evidence.cutoff)}`);
text("attribution", evidence.attribution);
text("provenance", JSON.stringify(evidence.provenance, null, 2));
if (evidence.recovery) {
  const note = document.createElement("p"); note.className = "scope";
  note.textContent = `${evidence.recovery.retained_controls} original results retained; ${evidence.recovery.recovered_controls} previously unmeasured controls recovered after a disk-space failure. Same fixed sample, keys and cutoff. This is not a fresh independent sample. Select a control to see which execution supplied it.`;
  $("bound").after(note);
}

for (const [index, item] of evidence.cases.entries()) {
  const option = document.createElement("option"); option.value = index;
  option.textContent = `${String(index + 1).padStart(2, "0")} / ${categoryName(item.category)} / source ${item.source_index}`;
  $("example").append(option);
}
const shortest = evidence.cases.map((c, i) => ({i, words: c.marked.words})).filter(c => c.words >= 100 && c.words <= 400).sort((a, b) => a.words - b.words)[0];
$("example").value = String(shortest?.i ?? 0);
function showExample() {
  const item = evidence.cases[Number($("example").value)];
  text("prompt", item.prompt);
  for (const kind of ["ordinary", "marked"]) {
    const row = item[kind];
    renderResponse($(kind + "-text"), row.text, $("format").value === "exact");
    text(kind + "-meta", `${row.words} words · ${row.completion === "eos" ? "Completed" : "Token limit reached; retained"}`);
    text(kind + "-decision", row.matching_flagged ? "Matching key flagged" : row.other_flagged ? "Other key flagged" : "Neither key flagged");
    text(kind + "-score", `Key A ${scoreText(row.scores[0])} / Key B ${scoreText(row.scores[1])} / Matching key ${row.key_index === 0 ? "A" : "B"}`);
  }
}
$("example").addEventListener("change", showExample);
$("format").addEventListener("change", showExample);
for (const button of document.querySelectorAll("button[data-side]")) {
  button.addEventListener("click", () => {
    document.querySelector(".pair").dataset.side = button.dataset.side;
    document.querySelectorAll("button[data-side]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
  });
}

for (const category of [...new Set(controls.map(r => r.category))].sort()) {
  const option = document.createElement("option"); option.value = category; option.textContent = categoryName(category);
  $("category").append(option);
}
let shown = [], active = 0;
function inspect(index) {
  active = index;
  const row = shown[index];
  $("control-detail").hidden = !row;
  if (!row) { $("grid").removeAttribute("aria-activedescendant"); return; }
  $("grid").setAttribute("aria-activedescendant", row.id);
  for (const option of $("grid").children) option.setAttribute("aria-selected", String(option.id === row.id));
  text("control-id", `${row.id} / source ${row.source_index}`);
  text("control-title", row.error ? "Unavailable" : row.flagged ? "Flagged" : "No flag");
  const origin = row.origin === "recovery" ? " · Recovery attempt after original disk guard" : row.origin === "retained" ? " · Retained from original attempt" : "";
  text("control-meta", `${categoryName(row.category)} · ${row.words} words${row.error ? " · " + row.error : ""}${origin}`);
  text("control-a", row.error ? "Unavailable" : scoreText(row.scores[0]));
  text("control-b", row.error ? "Unavailable" : scoreText(row.scores[1]));
}
function filter() {
  shown = controls.filter(r => ($("category").value === "all" || r.category === $("category").value) && ($("decision").value === "all" || ($("decision").value === "flagged" ? r.flagged && !r.error : r.error)));
  $("grid").replaceChildren();
  for (const [index, row] of shown.entries()) {
    const option = document.createElement("div"); option.id = row.id; option.setAttribute("role", "option");
    option.setAttribute("aria-label", `${row.id}, source ${row.source_index}, ${categoryName(row.category)}, ${row.words} words, ${row.error ? "unavailable" : row.flagged ? "flagged" : "no flag"}`);
    option.dataset.flag = String(Boolean(row.flagged)); option.dataset.error = String(Boolean(row.error));
    option.addEventListener("click", () => { inspect(index); $("grid").focus({preventScroll:true}); });
    $("grid").append(option);
  }
  $("empty").hidden = shown.length !== 0; $("grid").hidden = shown.length === 0;
  text("filter-count", `${shown.length} of ${controls.length} retained records`);
  inspect(0);
}
$("grid").addEventListener("keydown", event => {
  const columns = getComputedStyle($("grid")).gridTemplateColumns.split(" ").length;
  let next = active;
  if (event.key === "ArrowRight") next++;
  else if (event.key === "ArrowLeft") next--;
  else if (event.key === "ArrowDown") next += columns;
  else if (event.key === "ArrowUp") next -= columns;
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = shown.length - 1;
  else return;
  event.preventDefault(); inspect(Math.max(0, Math.min(shown.length - 1, next)));
  if (shown[active]) $(shown[active].id).scrollIntoView({block:"nearest", inline:"nearest"});
});
$("category").addEventListener("change", filter); $("decision").addEventListener("change", filter);
showExample(); filter();
