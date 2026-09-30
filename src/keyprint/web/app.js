function modelLabel(identity = {}) {
  if (identity.profile === "gguf-byte-bpe-v1-experimental")
    return "llama.cpp · experimental CPU";
  if (identity.profile === "portable-bytelevel-v1-experimental")
    return "Transformers · experimental CPU";
  return "Local model";
}

"use strict";
const $ = (id) => document.getElementById(id);
const replayMode = document.body?.dataset?.mode === "replay";
const tokenExplorer = typeof createTokenExplorer === "function"
  ? createTokenExplorer(document, {reducedMotion: () => matchMedia('(prefers-reduced-motion: reduce)').matches}) : null;
const fragment = new URLSearchParams(location.hash.slice(1));
const token =
  fragment.get("session") || sessionStorage.getItem("keyprint-session");
if (fragment.has("session")) {
  sessionStorage.setItem("keyprint-session", token);
  history.replaceState(null, "", location.pathname);
}
let busy = false,
  modelReady = false,
  connecting = false,
  experiment = null,
  recordedPrompt = null,
  editMeasurement = null,
  measuredText = null,
  pending = null,
  activeRequestId = null,
  activeAction = null,
  stopAvailable = false,
  stopRequested = false;
const ns = "http://www.w3.org/2000/svg";
const percent = (value) =>
  value == null ? "Unavailable" : (value * 100).toFixed(1) + "%";
// Inspection prefixes count Unicode code points, as Python strings do.
// UTF-16 offsets would shorten displayed prefixes after supplementary characters.
const characters = (text) => Array.from(text || "");
const rewriteExample = "Hi Maya, please review the draft by Friday at 09:30. Do not publish it before I approve the final version. If approval has not arrived by the deadline, postpone publication and send me an update. Keep the backup until you have verified that the restore succeeded.";
let taskMode = "generate";
const drafts = { generate: $("prompt").value, rewrite: rewriteExample };

function selectTaskMode(mode) {
  if (!["generate", "rewrite"].includes(mode)) return;
  drafts[taskMode] = $("prompt").value;
  taskMode = mode;
  $("task-mode").value = mode;
  $("prompt").value = drafts[mode];
  $("prompt").maxLength = mode === "rewrite" ? 8000 : 6000;
  $("input-label").textContent = mode === "rewrite" ? "Your original text" : "Your prompt";
  $("input-summary").textContent = mode === "rewrite" ? "Edit your text or response limit" : "Change prompt or response limit";
  $("generate-presets").hidden = mode === "rewrite";
  $("rewrite-controls").hidden = mode !== "rewrite";
  $("run-description").textContent = mode === "rewrite"
    ? "One real local rewrite beside your unchanged original. Compare meaning and check the flagged details."
    : "Two independent samples from the same model and prompt. One ordinary, one marked.";
  if (!busy) $("generate").textContent = mode === "rewrite" ? "Rewrite this text ↗" : "Generate both versions ↗";
  syncPromptPreview();
}

function syncRecipe() {
  if (!$("recipe")) return;
  const selected = $("recipe-backend")?.value;
  const [constructor,path] = ({
    mlx: ['from_mlx','path/to/qwen3-8b-4bit'],
    transformers: ['from_transformers','path/to/qualified-local-model'],
    llama_cpp: ['from_llama_cpp','path/to/qualified-model.gguf'],
  })[selected] || ['from_mlx','path/to/qwen3-8b-4bit'];
  if ($("backend-install")) $("backend-install").textContent =
    `python -m pip install '.[${({transformers:'transformers',llama_cpp:'llama-cpp'})[selected] || 'mlx'}]'`;
  $("recipe").textContent = `from keyprint import Keyprint\n\nwith Keyprint.${constructor}(${JSON.stringify(path)}, key=Keyprint.new_key()) as wm:\n    pair = wm.compare(${JSON.stringify($("prompt").value)}, max_tokens=${Number($("cap").value) || 192})\n    print(pair.marked)\n    pair.export("my-demo")`;
}
function syncPromptPreview() {
  syncRecipe();
  if (replayMode && recordedPrompt !== null && experiment) {
    $("status").textContent = $("prompt").value !== recordedPrompt
      ? `Your new prompt has not been run. The recorded outputs below still answer: “${recordedPrompt}”`
      : "Recorded SDK run. Explore its exact outputs and prefix measurements. No model runs in this browser.";
  }
  $("prompt-preview").textContent = $("prompt").value.trim() ||
    (taskMode === "rewrite" ? "Paste your text to try a local rewrite." : "Enter a prompt to generate your own pair.");
}

function readResponse(condition) {
  if (!["ordinary", "marked"].includes(condition)) return;
  $("outputs").setAttribute("data-focus", condition);
  for (const name of ["ordinary", "marked"])
    $("read-" + name).setAttribute("aria-pressed", String(condition === name));
}

function setBusy(value) {
  busy = value;
  for (const id of ["generate", "inspect", "half", "restore"])
    $(id).disabled = value || !modelReady || (id !== "generate" && !experiment);
  for (const control of [$("prompt"), $("cap"), $("task-mode"), $("preserve"), $("rewrite-example"), ...document.querySelectorAll("[data-prompt]")])
    control.disabled = value;
  $("outputs").setAttribute("aria-busy", String(value));
  $("generate").textContent = value
    ? "Running local experiment…"
    : taskMode === "rewrite" ? "Rewrite this text ↗" : "Generate both versions ↗";
  if (!value) {
    if (stopRequested && ["stop", "stop-edit", ""].includes(document.activeElement?.id || "")) {
      const target = activeAction === "inspect" ? $("inspect")
        : $("prompt-controls").open ? $("generate")
        : $("prompt-controls").querySelector("summary");
      target.focus();
    }
    activeRequestId = null;
    stopAvailable = false;
    stopRequested = false;
  }
  updateStopControls();
}

function updateStopControls() {
  for (const id of ["stop", "stop-edit"]) {
    $(id).hidden = !busy || (id === "stop-edit" && activeAction !== "inspect");
    $(id).disabled = !busy || !stopAvailable || stopRequested ||
      (id === "stop-edit" && activeAction !== "inspect");
    $(id).textContent = stopRequested ? "Stopping…" : "Stop";
  }
}

async function api(path, body, id, signal) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: "Bearer " + token,
      ...(body
        ? { "Content-Type": "application/json", "Idempotency-Key": id }
        : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
    ...(signal ? { signal } : {}),
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error?.message || "Local request failed");
    error.settled = response.status !== 409;
    error.status = response.status;
    throw error;
  }
  return data;
}

function watchProgress(status) {
  const controller = new AbortController();
  let stopped = false,
    timer;
  const stages = {
    generating_ordinary: "Generating the ordinary response",
    inspecting_ordinary: "Measuring the ordinary response",
    generating_marked: "Generating the watermarked response",
    inspecting_marked: "Measuring the watermarked response",
    inspecting_edit: "Measuring your edited text",
    rewriting: "Rewriting your text locally",
    inspecting_source: "Measuring your unchanged original",
  };
  async function poll() {
    try {
      const progress = await api("/api/progress", undefined, undefined, controller.signal);
      if (!stopped && progress.active && progress.request_id === activeRequestId) {
        stopAvailable = true;
        stopRequested ||= progress.cancellation_requested;
        updateStopControls();
        if (stopRequested)
          status.textContent = "Stopping after the current model step or measurement. Previous results stay visible.";
        else if (stages[progress.stage])
          status.textContent = `${stages[progress.stage]} · ${progress.seconds.toFixed(0)}s elapsed. Work continues locally.`;
      }
    } catch {
      // Losing a progress update must not retry or replace the model request.
    }
    if (!stopped) timer = setTimeout(poll, 1000);
  }
  timer = setTimeout(poll, 500);
  return () => {
    stopped = true;
    clearTimeout(timer);
    controller.abort();
  };
}

async function stopExperiment() {
  if (!busy || !stopAvailable || stopRequested || !activeRequestId) return;
  const id = activeRequestId;
  const status = $(activeAction === "inspect" ? "edit-status" : "status");
  stopRequested = true;
  updateStopControls();
  status.textContent = "Stopping after the current model step or measurement. Previous results stay visible.";
  try {
    await api("/api/cancel", {}, id);
    // The original request/session polling owns completion. A stop request
    // cannot unlock controls or replace a result that already finished.
  } catch (error) {
    if (busy && activeRequestId === id) {
      stopRequested = false;
      updateStopControls();
      status.textContent = "Could not request a stop. You can try Stop again or refresh to reconnect.";
    }
  }
}

function report() {
  const data = {
    experiment,
    edited: editMeasurement ? { text: measuredText, ...editMeasurement } : null,
  };
  $("report").textContent = JSON.stringify(data, null, 2);
  $("download").disabled = !experiment;
  return data;
}

function element(tag, attributes, text) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  $("chart").append(node);
  return node;
}

function chart() {
  const svg = $("chart");
  svg.replaceChildren();
  element(
    "title",
    { id: "chart-title" },
    "Literal watermark diagnostic by text prefix",
  );
  const original = experiment?.outputs.marked;
  const edited = editMeasurement?.inspection;
  element(
    "desc",
    { id: "chart-description" },
    original
      ? `Original matching key: ${percent(original.inspection.fraction)}. Other key: ${percent(original.inspection.control_fraction)}. ${edited ? "Edited matching key: " + percent(edited.fraction) + "." : ""} Uncalibrated observed bit fractions, not detection confidence.`
      : "No measurements yet.",
  );
  const series = original?.inspection.series || [];
  const max = Math.max(characters(original?.text).length, characters(measuredText).length, 1);
  const values = [
    ...series.flatMap((p) => [p.matching, p.control]),
    ...(edited?.series || []).map((p) => p.matching),
  ].filter((v) => v != null);
  const low = Math.max(
    0,
    Math.floor((Math.min(0.4, ...values) - 0.03) * 10) / 10,
  );
  const high = Math.min(
    1,
    Math.ceil((Math.max(0.6, ...values) + 0.03) * 10) / 10,
  );
  const x = (n) => 48 + (n / max) * 496,
    y = (v) => 225 - ((v - low) / (high - low)) * 190;
  for (let i = 0; i <= 4; i++) {
    const value = low + ((high - low) * i) / 4;
    element("line", {
      x1: 48,
      x2: 544,
      y1: y(value),
      y2: y(value),
      class: "grid",
    });
    element(
      "text",
      { x: 36, y: y(value) + 4, "text-anchor": "end" },
      Math.round(value * 100) + "%",
    );
  }
  element("line", {
    x1: 48,
    x2: 544,
    y1: y(0.5),
    y2: y(0.5),
    class: "baseline",
  });
  element("text", { x: 48, y: 252 }, "0");
  element(
    "text",
    { x: 544, y: 252, "text-anchor": "end" },
    max + " characters",
  );
  function curve(points, field, style) {
    let active = false,
      path = "";
    for (const point of points) {
      const value = point[field];
      if (value == null) {
        active = false;
        continue;
      }
      path += `${active ? "L" : "M"}${x(point.characters)},${y(value)} `;
      active = true;
    }
    element("path", { d: path, class: "curve curve-" + style });
  }
  curve(series, "matching", "original");
  curve(series, "control", "control");
  if (edited) curve(edited.series, "matching", "edited");
  const point = series[Number($("scrub").value)];
  if (point) {
    element("line", {
      x1: x(point.characters),
      x2: x(point.characters),
      y1: 28,
      y2: 228,
      class: "cursor",
    });
    if (point.matching != null)
      element("circle", {
        cx: x(point.characters),
        cy: y(point.matching),
        r: 4,
        fill: "#a3452b",
      });
    if (point.control != null)
      element("circle", {
        cx: x(point.characters),
        cy: y(point.control),
        r: 4,
        fill: "#576b83",
      });
    $("prefix").textContent = characters(original.text).slice(0, point.characters).join("");
    $("prefix-readout").textContent =
      `${point.characters} characters · matching ${percent(point.matching)} · other ${percent(point.control)}`;
  }
}

function showMetrics(inspection, label) {
  $("fraction").textContent =
    inspection.fraction == null ? "—" : percent(inspection.fraction);
  $("events").textContent = inspection.events ?? "—";
  $("bits").textContent = inspection.trials ?? "—";
  $("measurement-status").textContent =
    `${label}. Uncalibrated diagnostic; no detection verdict.`;
}

function markDirty() {
  const dirty =
    $("edited").value !== (measuredText ?? experiment?.outputs.marked.text);
  $("edit-status").textContent = dirty
    ? "Edits not measured yet. Select “Update signal” to inspect this text."
    : "Displayed measurements match this text.";
  if (dirty)
    $("measurement-status").textContent =
      "Showing the last measured text, not your pending edits.";
  else if (experiment)
    showMetrics(editMeasurement?.inspection ?? experiment.outputs.marked.inspection,
      editMeasurement ? "Last measured edit" : "Original marked response");
}

function restoreLimit(limit) {
  if (Number.isInteger(limit) && limit >= 32 && limit <= 1024) {
    const cap = $("cap");
    const value = String(limit);
    if (![...cap.options].some((option) => option.value === value)) {
      cap.add(new Option(`${limit} tokens`, value));
    }
    cap.value = value;
  }
}

function restoreRequest(session) {
  const request = session.last_attempt?.request;
  if (["generate", "rewrite"].includes(request?.action)) {
    selectTaskMode(request.action);
    $("prompt").value = request.text;
    if (request.action === "rewrite") $("preserve").value = (request.preserve || []).join("\n");
    syncPromptPreview();
    restoreLimit(request.max_tokens);
  }
}

function renderOutputs() {
  if (!experiment) return;
  $("wording-toggle").disabled = false;
  const differences = $("wording-toggle").getAttribute("aria-pressed") === "true";
  $("wording-note").hidden = !differences;
  $("reading-mode").disabled = differences;
  if (differences) {
    $("wording-note").textContent = renderWordingPair($("ordinary-text"), $("marked-text"),
      experiment.outputs.ordinary.text, experiment.outputs.marked.text);
    return;
  }
  for (const condition of ["ordinary", "marked"])
    renderResponse($(condition + "-text"), experiment.outputs[condition].text,
      $("reading-mode").value === "exact");
}

function restoreInspection(session) {
  if (session.edited) {
    editMeasurement = session.edited.result;
    measuredText = session.edited.text;
    $("edited").value = measuredText;
    showMetrics(editMeasurement.inspection, "Restored measured edit");
    $("edit-status").textContent = "Previous measured edit restored. No new inspection started.";
  }
  const attempt = session.last_attempt;
  if (attempt?.request?.action === "inspect") {
    $("edited").value = attempt.request.text;
    if (attempt.http_status !== 200) {
      markDirty();
      $("edit-status").textContent = attempt.http_status == null
        ? "Reconnecting to the inspection of this text. Previous measurements remain visible."
        : attempt.http_status === 410
          ? "Inspection stopped. Your text is restored; measurements still describe the last successful text."
          : "This inspection failed. Your text is restored; measurements still describe the last successful text. No automatic retry.";
    }
  }
  chart();
  report();
}

function showGeneration(data, retained = false) {
  const rewriting = data.action === "rewrite";
  if (retained) {
    const prompt = rewriting ? data.outputs.ordinary.text : $("prompt").value;
    selectTaskMode(rewriting ? "rewrite" : "generate");
    $("prompt").value = prompt;
    if (rewriting) $("preserve").value = (data.preserve || []).join("\n");
  }
  syncPromptPreview();
  experiment = data;
  if (["mlx", "transformers", "llama_cpp"].includes(data.backend)) {
    $("recipe-backend").value = data.backend;
    syncRecipe();
  }
  tokenExplorer?.show(data);
  if (retained) restoreLimit(data.max_tokens);
  editMeasurement = null;
  measuredText = null;
  $("ordinary-heading").textContent = rewriting ? "Your original" : "Ordinary";
  $("marked-heading").textContent = rewriting ? "Local rewrite" : "With Keyprint";
  $("read-ordinary").textContent = rewriting ? "Your original" : "Ordinary";
  $("read-marked").textContent = rewriting ? "Local rewrite" : "With Keyprint";
  $("ordinary-text").setAttribute("aria-label", rewriting ? "Your original text" : "Ordinary response");
  $("marked-text").setAttribute("aria-label", rewriting ? "Local rewrite candidate" : "Watermarked response");
  showRewriteChecks(data.rewrite);
  for (const condition of ["ordinary", "marked"]) {
    const result = data.outputs[condition];
    $(condition + "-text").classList.remove("placeholder");
    $(condition + "-meta").textContent = result.completion === "source" ? "Your text · unchanged" :
      `${result.usage?.completion_tokens ?? "?"} tokens · ${result.completion === "eos" ? "complete" : result.completion === "length" ? "limit reached" : "completion unavailable"}` +
      (result.timing
        ? ` · ${result.timing.generation_seconds.toFixed(1)}s generation · ${result.timing.inspection_seconds.toFixed(1)}s inspection`
        : "");
  }
  renderOutputs();
  $("edited").disabled = false;
  $("edited").value = data.outputs.marked.text;
  $("scrub").disabled = false;
  $("scrub").max = Math.max(
    0,
    data.outputs.marked.inspection.series.length - 1,
  );
  $("scrub").value = $("scrub").max;
  const capped = Object.values(data.outputs).filter(
    (result) => result.completion === "length",
  ).length;
  const outcome = capped && rewriting
    ? `The rewrite reached the token limit. ${data.max_tokens >= 1024 ? "Shorten the source" : "Choose a larger limit or shorten the source"}, then start a new rewrite.`
    : capped
    ? `${capped === 2 ? "Both responses" : "One response"} reached the token limit. ${data.max_tokens >= 1024 ? "Ask for a shorter answer, then generate a new pair." : "Choose a larger limit or ask for a shorter answer, then generate a new pair."}`
    : rewriting ? "Local rewrite finished. Compare its meaning with your original."
    : Object.values(data.outputs).every((result) => result.completion === "eos")
      ? "Both responses finished."
      : "Completion state unavailable; inspect the exported report.";
  $("status").textContent =
    `${retained ? "Previous live run restored" : "Live run finished"} · ${data.seconds.toFixed(1)}s generation and inspection. ${outcome}` +
    (rewriting ? data.rewrite.status === "failed_checks"
      ? " Some rewrite checks failed. See “Compare before reuse”."
      : " Literal checks passed; meaning is not verified."
      : " Independent samples; differences alone are not a quality test.");
  $("edit-status").textContent =
    "Change words, remove a sentence, or paste text. Then update the signal.";
  showMetrics(data.outputs.marked.inspection, "Original marked response");
  chart();
  report();
}

function showRewriteChecks(rewrite) {
  $("rewrite-review").hidden = !rewrite;
  $("rewrite-issues").replaceChildren();
  if (!rewrite) return;
  const labels = {
    complete: "The rewrite reached its token limit or did not finish.",
    nonempty: "The rewrite is empty.",
    canonical_changed: "Only whitespace changed; this is not a paraphrase.",
    word_sequence_changed: "The word sequence did not change.",
    numbers_preserved: "Numbers, times or their occurrence counts changed.",
    weekday_names_preserved: "Weekday names or their occurrence counts changed.",
    urls_preserved: "Links or their occurrence counts changed.",
    emails_preserved: "Email addresses or their occurrence counts changed.",
    no_new_escaped_line_breaks: "Literal line-break escape sequences were introduced.",
    protected_literals_preserved: "A protected phrase or its occurrence count changed.",
  };
  $("rewrite-review-summary").textContent = rewrite.status === "failed_checks"
    ? "Details changed or the rewrite did not satisfy its basic checks. Review the candidate above before reusing it."
    : "Literal checks passed. They cannot tell whether the meaning stayed the same.";
  for (const [check, label] of Object.entries(labels)) {
    if (rewrite.checks[check] === true) continue;
    const item = document.createElement("li");
    item.textContent = label;
    $("rewrite-issues").append(item);
  }
}

async function run(action) {
  if (replayMode) {
    $("setup").open = true;
    syncRecipe();
    $("setup").scrollIntoView({behavior: "smooth", block: "start"});
    $("copy-recipe").focus({preventScroll: true});
    return;
  }
  if (busy || !modelReady) return;
  const text = action !== "inspect" ? $("prompt").value : $("edited").value;
  if (!text.trim()) {
    $(action !== "inspect" ? "status" : "edit-status").textContent =
      "Enter some text first.";
    return;
  }
  const body = { action, text, max_tokens: Number($("cap").value) };
  if (action === "rewrite") body.preserve = $("preserve").value.split("\n").filter((line) => line.trim());
  // A transport retry for identical input reuses the same ID, so it cannot
  // silently generate a second pair if the first request completed unseen.
  const serialized = JSON.stringify(body);
  if (!pending || pending.serialized !== serialized)
    pending = { serialized, id: crypto.randomUUID() };
  activeRequestId = pending.id;
  activeAction = action;
  stopAvailable = false;
  stopRequested = false;
  setBusy(true);
  const status = action !== "inspect" ? $("status") : $("edit-status");
  status.classList.remove("error");
  status.textContent =
    action === "generate"
      ? "Generating two real responses, then measuring both. The previous result stays visible until completion."
      : action === "rewrite" ? "Rewriting your source locally, then measuring both texts. Your previous result stays visible."
      : "Replaying edited text with the SDK…";
  const stopProgress = watchProgress(status);
  try {
    const data = await api("/api/experiment", body, pending.id);
    stopProgress();
    pending = null;
    if (action !== "inspect") {
      showGeneration(data);
    } else {
      editMeasurement = data;
      measuredText = text;
      showMetrics(data.inspection, "Last measured edit");
      status.textContent = `Edit inspected in ${data.seconds.toFixed(1)}s. The original remains visible for comparison.`;
    }
    chart();
    report();
    if (action === "inspect" && $("edited").value !== text) markDirty();
  } catch (error) {
    stopProgress();
    if (error.settled) pending = null;
    if (action === "inspect") markDirty();
    status.classList.toggle("error", error.status !== 410);
    status.textContent = error.status === 410
      ? action === "inspect"
        ? "Inspection stopped. Your edits remain; previous measurements are unchanged."
        : "Stopped. Previous completed results are unchanged. Generate again when ready."
      : error.message + " No automatic retry.";
  } finally {
    stopProgress();
    setBusy(false);
  }
}

$("generate").addEventListener("click", () => run(taskMode));
$("task-mode").addEventListener("change", () => {
  selectTaskMode($("task-mode").value);
  $("prompt-controls").open = true;
});
$("rewrite-example").addEventListener("click", () => {
  $("prompt").value = rewriteExample;
  $("preserve").value = "Maya";
  syncPromptPreview();
  $("prompt").focus();
});
$("stop").addEventListener("click", stopExperiment);
$("stop-edit").addEventListener("click", stopExperiment);
$("reading-mode").addEventListener("change", renderOutputs);
$("wording-toggle").addEventListener("click", () => {
  const active = $("wording-toggle").getAttribute("aria-pressed") !== "true";
  $("wording-toggle").setAttribute("aria-pressed", String(active));
  $("wording-toggle").textContent = active ? "Hide differences" : "Highlight differences";
  renderOutputs();
});
$("inspect").addEventListener("click", () => run("inspect"));
$("edited").addEventListener("input", markDirty);
$("scrub").addEventListener("input", chart);
function scrubChart(event) {
  if (!experiment) return;
  const bounds = $("chart").getBoundingClientRect();
  const position =
    (((event.clientX - bounds.left) / bounds.width) * 560 - 48) / 496;
  const length = Math.max(
    characters(experiment.outputs.marked.text).length,
    characters(measuredText).length,
    1,
  );
  const points = experiment.outputs.marked.inspection.series;
  if (!points.length) return;
  let closest = 0;
  points.forEach((point, index) => {
    if (
      Math.abs(point.characters - position * length) <
      Math.abs(points[closest].characters - position * length)
    )
      closest = index;
  });
  $("scrub").value = closest;
  chart();
}
$("chart").addEventListener("pointerdown", (event) => {
  $("chart").setPointerCapture(event.pointerId);
  scrubChart(event);
});
$("chart").addEventListener("pointermove", (event) => {
  if ($("chart").hasPointerCapture(event.pointerId)) scrubChart(event);
});
$("half").addEventListener("click", () => {
  const text = characters($("edited").value);
  const cut = text.lastIndexOf(" ", Math.floor(text.length / 2));
  $("edited").value = text.slice(0, cut > 0 ? cut : Math.ceil(text.length / 2)).join("");
  run("inspect");
});
$("restore").addEventListener("click", () => {
  $("edited").value = experiment.outputs.marked.text;
  editMeasurement = null;
  measuredText = null;
  showMetrics(experiment.outputs.marked.inspection, "Original marked response");
  chart();
  report();
  markDirty();
});
document.querySelectorAll("[data-prompt]").forEach((button) =>
  button.addEventListener("click", () => {
    $("prompt").value = button.dataset.prompt;
    syncPromptPreview();
    $("prompt").focus();
  }),
);
$("prompt").addEventListener("input", syncPromptPreview);
$("read-ordinary").addEventListener("click", () => readResponse("ordinary"));
$("read-marked").addEventListener("click", () => readResponse("marked"));
$("download").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(report(), null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "keyprint-experiment.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

chart();
setBusy(false);
async function connect() {
  if (connecting || busy) return;
  connecting = true;
  modelReady = false;
  setBusy(false);
  $("reconnect").hidden = true;
  $("status").classList.remove("error");
  try {
    if (!token)
      throw new Error(
        "Open the complete session URL printed by “keyprint playground”",
      );
    let session = await api("/api/session");
    while (session.model?.status === "loading") {
      modelReady = false;
      setBusy(false);
      $("model-label").textContent = "Loading the local model";
      $("status").textContent = `Loading model locally · ${session.model.seconds.toFixed(0)}s elapsed. You can edit the prompt while it loads. No generation has started.`;
      await new Promise((resolve) => setTimeout(resolve, 1000));
      session = await api("/api/session");
    }
    if (session.model?.status === "failed") {
      modelReady = false;
      $("model-label").textContent = "Model unavailable";
      throw new Error(session.model.message);
    }
    modelReady = true;
    $("model-label").textContent = modelLabel(session.identity);
    if (session.running) {
      if (session.latest) {
        $("prompt").value = session.prompt;
        showGeneration(session.latest, true);
        restoreInspection(session);
      }
      restoreRequest(session);
      activeRequestId = session.last_attempt?.request_id;
      activeAction = session.last_attempt?.action;
      stopAvailable = true;
      stopRequested = session.last_attempt?.cancellation_requested || false;
      setBusy(true);
      const progressStatus = $(activeAction === "inspect" ? "edit-status" : "status");
      progressStatus.textContent =
        "Reconnecting to the running experiment. No new generation started.";
      const stopProgress = watchProgress(progressStatus);
      try {
        while (session.running) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          session = await api("/api/session");
        }
      } finally {
        stopProgress();
        setBusy(false);
      }
    }
    if (session.latest) {
      $("prompt").value = session.prompt;
      showGeneration(session.latest, true);
      restoreInspection(session);
      setBusy(false);
      if (session.last_attempt?.http_status >= 400) {
        restoreRequest(session);
        const cancelled = session.last_attempt.http_status === 410;
        $("status").textContent = cancelled
          ? "The latest experiment was stopped. Showing the previous completed run. No new work started."
          : "The latest attempt failed. Showing the previous completed run; no automatic retry.";
        $("status").classList.toggle("error", !cancelled);
      }
      return;
    }
    if (session.last_attempt) {
      restoreRequest(session);
      if (session.last_attempt.http_status === 410) {
        setBusy(false);
        $("status").textContent = "The previous experiment was stopped. No new work started. Generate again when ready.";
        return;
      }
      throw new Error(
        "The previous attempt did not produce a completed pair. No automatic retry; start a new experiment explicitly.",
      );
    }
    setBusy(false);
    await run(taskMode);
  } catch (error) {
    setBusy(false);
    $("status").textContent = error.message;
    $("status").classList.add("error");
    $("outputs").setAttribute("aria-busy", "false");
    $("reconnect").hidden = false;
  } finally {
    connecting = false;
  }
}
$("reconnect").addEventListener("click", () => replayMode ? loadReplay() : connect());
$("cap").addEventListener("change", syncRecipe);
$("recipe-backend").addEventListener("change", syncRecipe);
$("gallery-custom").addEventListener("click", () => {
  $("prompt-controls").open = true;
  $("prompt").focus();
  $("prompt-controls").scrollIntoView({block: "start"});
});
$("copy-recipe").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("recipe").textContent);
    $("copy-status").textContent = "Copied. Run from your local SDK environment with your model path.";
  } catch {
    $("copy-status").textContent = "Copy unavailable. Select the Python above to copy it.";
  }
});

function showRecording(data) {
  if (data.schema !== "keyprint-comparison-v1" || typeof data.prompt !== "string")
    throw new Error("Unsupported recording format.");
    $("prompt").value = data.prompt;
    recordedPrompt = data.prompt;
    $("prompt-controls").open = false;
    $("intro-copy").textContent = "Read two real responses. Follow the measured pattern. Then build your own.";
    $("explore-link").textContent = "Explore the recorded signal ↓";
    $("explore-title").textContent = "Follow the signal through the text.";
    $("edited-legend").hidden = true;
    showGeneration(data.experiment, true);
    $("outputs").setAttribute("aria-busy", "false");
    $("status").textContent = "Recorded SDK run. These exact outputs and prefix measurements were produced by a local model. No generation runs in this browser." +
      (Object.values(data.experiment.outputs).some(r => r.completion === "length") ? " A response reached its token limit; the original ending is retained." : "") +
      (data.experiment.outputs.ordinary.text === data.experiment.outputs.marked.text ? " Both samples are identical; that is a valid possible outcome." : "");
    $("generate").disabled = false;
    $("generate").textContent = "Use this prompt locally";
    $("run-description").textContent = "Change the prompt, then copy the matching Python recipe. Live generation requires your local SDK.";
    $("edited").readOnly = true;
    $("edit-label").textContent = "Recorded marked response";
    $("edit-status").textContent = "Scrub the chart to explore actual prefix measurements. Arbitrary edits require the live local playground.";
    for (const id of ["half", "restore", "inspect", "stop-edit"]) $(id).hidden = true;
    syncRecipe();
}

async function loadReplay() {
  // Static replay never sends a prompt, key or request to a model endpoint.
  $("reconnect").hidden = true;
  $("experience-label").textContent = "Recorded SDK run · explore without an install";
  $("model-label").textContent = "Recorded model output";
  $("status").textContent = "Loading the recorded SDK run…";
  try {
    const response = await fetch(document.body?.dataset?.gallery === "true" ? "./gallery.json" : "./replay.json");
    if (!response.ok) throw new Error("The recording could not be loaded.");
    const data = await response.json();
    if (document.body?.dataset?.gallery === "true") {
      mountGallery(document, data, showRecording);
    } else {
      showRecording(data);
    }
  } catch (error) {
    $("status").textContent = "Recording unavailable. " + error.message;
    $("reconnect").textContent = "Reload recording";
    $("reconnect").hidden = false;
  }
}
replayMode ? loadReplay() : connect();
