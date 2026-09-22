"use strict";

// Render only the SDK's exact committed tokens. No word-level attribution,
// invented alternatives, model requests, or reconstructed random draws.
function createTokenExplorer(doc, { reducedMotion = () => false } = {}) {
  const get = id => doc.getElementById(id);
  let experiment = null, steps = [], position = 0, timer = null;
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    get('trace-play').textContent = 'Replay tokens';
    get('trace-play').setAttribute('aria-pressed', 'false');
  }
  function select(index) {
    if (!steps.length) return;
    position = Math.max(0, Math.min(index, steps.length - 1));
    const point = steps[position];
    get('trace-position').value = String(position);
    get('trace-prefix').textContent = steps.slice(0, position + 1).map(p => p.text).join('') || '(No visible text yet)';
    get('trace-readout').textContent = `Token ${position + 1} of ${steps.length} · ${point.kind === 'control' ? 'end/control token' : point.kind === 'pending_bytes' ? 'character continues in another token' : 'selected text'}`;
    get('trace-details').textContent = `Token ID ${point.token_id} · bytes ${point.bytes_hex ?? 'control'} · character offsets ${point.start}–${point.end}`;
    Array.from(get('trace-tokens').children).forEach((button, i) => {
      button.setAttribute('aria-pressed', String(i === position));
      button.classList.toggle('trace-future', i > position);
    });
  }
  function loadCondition() {
    stop();
    steps = experiment?.outputs[get('trace-condition').value]?.trace || [];
    get('trace-tokens').replaceChildren();
    get('choices').hidden = steps.length === 0;
    if (!steps.length) return;
    for (const step of steps) {
      const button = doc.createElement('button');
      button.type = 'button';
      const label = step.text || (step.kind === 'control' ? 'end' : `bytes ${step.bytes_hex}`);
      button.textContent = label;
      button.setAttribute('aria-label', `Token ${step.index + 1}: ${label}`);
      button.addEventListener('click', () => { stop(); select(step.index); });
      get('trace-tokens').append(button);
    }
    get('trace-position').max = String(steps.length - 1);
    select(steps.length - 1);
  }
  get('trace-condition').addEventListener('change', loadCondition);
  get('trace-position').addEventListener('input', () => { stop(); select(Number(get('trace-position').value)); });
  get('trace-play').addEventListener('click', () => {
    if (timer !== null) { stop(); return; }
    if (reducedMotion()) { select(position === steps.length - 1 ? 0 : position + 1); return; }
    select(0);
    get('trace-play').textContent = 'Pause';
    get('trace-play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      select(position + 1);
      if (position === steps.length - 1) stop();
    }, 110);
  });
  return { show(data) { experiment = data; loadCondition(); }, stop };
}
if (typeof module !== 'undefined') module.exports = { createTokenExplorer };
