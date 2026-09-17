"use strict";
const $ = (id) => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
const token =
  fragment.get("session") || sessionStorage.getItem("keyprint-session");
if (fragment.has("session")) {
  sessionStorage.setItem("keyprint-session", token);
  history.replaceState(null, "", location.pathname);
}
let busy = false,
  experiment = null,
  editMeasurement = null,
  measuredText = null,
  pending = null;
const ns = "http://www.w3.org/2000/svg";
const percent = (value) =>
  value == null ? "Unavailable" : (value * 100).toFixed(1) + "%";

function setBusy(value) {
  busy = value;
  for (const id of ["generate", "inspect", "half", "restore"])
    $(id).disabled = value || (id !== "generate" && !experiment);
  $("outputs").setAttribute("aria-busy", String(value));
  $("generate").textContent = value
    ? "Running local experiment…"
    : "Generate both versions ↗";
}

async function api(path, body, id) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: "Bearer " + token,
      ...(body
        ? { "Content-Type": "application/json", "Idempotency-Key": id }
        : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error?.message || "Local request failed");
    error.settled = response.status !== 409;
    throw error;
  }
  return data;
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
  const max = Math.max(original?.text.length || 1, measuredText?.length || 1);
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
    $("prefix").textContent = original.text.slice(0, point.characters);
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
}

function showGeneration(data, retained = false) {
  experiment = data;
  editMeasurement = null;
  measuredText = null;
  for (const condition of ["ordinary", "marked"]) {
    const result = data.outputs[condition];
    $(condition + "-text").textContent =
      result.text || "(Empty model response)";
    $(condition + "-text").classList.remove("placeholder");
    $(condition + "-meta").textContent =
      `${result.usage?.completion_tokens ?? "?"} tokens · ${result.completion === "eos" ? "complete" : "limit reached"}`;
  }
  $("edited").disabled = false;
  $("edited").value = data.outputs.marked.text;
  $("scrub").disabled = false;
  $("scrub").max = Math.max(
    0,
    data.outputs.marked.inspection.series.length - 1,
  );
  $("scrub").value = $("scrub").max;
  $("status").textContent =
    `${retained ? "Previous live run restored" : "Live run complete"} · ${data.seconds.toFixed(1)}s generation and inspection. Independent samples; differences alone are not a quality test.`;
  $("edit-status").textContent =
    "Change words, remove a sentence, or paste text. Then update the signal.";
  showMetrics(data.outputs.marked.inspection, "Original marked response");
  chart();
  report();
}

async function run(action) {
  if (busy) return;
  const text = action === "generate" ? $("prompt").value : $("edited").value;
  if (!text.trim()) {
    $(action === "generate" ? "status" : "edit-status").textContent =
      "Enter some text first.";
    return;
  }
  const body = { action, text, max_tokens: Number($("cap").value) };
  // A transport retry for identical input reuses the same ID, so it cannot
  // silently generate a second pair if the first request completed unseen.
  const serialized = JSON.stringify(body);
  if (!pending || pending.serialized !== serialized)
    pending = { serialized, id: crypto.randomUUID() };
  setBusy(true);
  const status = action === "generate" ? $("status") : $("edit-status");
  status.classList.remove("error");
  status.textContent =
    action === "generate"
      ? "Generating two real responses, then measuring both. The previous result stays visible until completion."
      : "Replaying edited text with the SDK…";
  try {
    const data = await api("/api/experiment", body, pending.id);
    pending = null;
    if (action === "generate") {
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
    if (error.settled) pending = null;
    status.classList.add("error");
    status.textContent = error.message + " No automatic retry.";
  } finally {
    setBusy(false);
  }
}

$("generate").addEventListener("click", () => run("generate"));
$("inspect").addEventListener("click", () => run("inspect"));
$("edited").addEventListener("input", markDirty);
$("scrub").addEventListener("input", chart);
function scrubChart(event) {
  if (!experiment) return;
  const bounds = $("chart").getBoundingClientRect();
  const position =
    (((event.clientX - bounds.left) / bounds.width) * 560 - 48) / 496;
  const length = Math.max(
    experiment.outputs.marked.text.length,
    measuredText?.length || 1,
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
  const text = $("edited").value;
  const cut = text.lastIndexOf(" ", Math.floor(text.length / 2));
  $("edited").value = text.slice(0, cut > 0 ? cut : Math.ceil(text.length / 2));
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
    $("prompt").focus();
  }),
);
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
(async () => {
  try {
    if (!token)
      throw new Error(
        "Open the complete session URL printed by “keyprint playground”",
      );
    const session = await api("/api/session");
    $("model-label").textContent = session.identity.profile?.startsWith(
      "portable",
    )
      ? "Transformers · experimental CPU"
      : "Qwen3-8B · local MLX";
    if (session.latest) {
      $("prompt").value = session.prompt;
      showGeneration(session.latest, true);
      setBusy(false);
      return;
    }
    setBusy(false);
    await run("generate");
  } catch (error) {
    $("status").textContent = error.message;
    $("status").classList.add("error");
    $("outputs").setAttribute("aria-busy", "false");
  }
})();
