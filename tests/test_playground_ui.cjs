// Browser-independent regression tests for the UI's measurement coordinates and
// edit state. Rendered behavior and real inference are verified separately.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function ui() {
  const nodes = new Map();
  function node() {
    return {
      textContent: '', value: '0', children: [], attributes: {}, listeners: {},
      classList: { add() {}, remove() {}, toggle() {} },
      addEventListener(event, fn) { this.listeners[event] = fn; },
      setAttribute(key, value) { this.attributes[key] = String(value); },
      replaceChildren() { this.children = []; },
      append(child) { this.children.push(child); },
      getBoundingClientRect() { return { left: 0, width: 560 }; },
    };
  }
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: get, createElementNS: () => node(), querySelectorAll: () => [] },
    location: { hash: '' }, sessionStorage: { getItem: () => null }, URLSearchParams,
  });
  // No token: startup reports the normal session instruction without network IO.
  vm.runInContext(fs.readFileSync(require.resolve('../src/keyprint/web/app.js'), 'utf8'), context);
  const run = code => vm.runInContext(code, context);
  return { get, run, set: (name, value) => run(`${name} = ${JSON.stringify(value)}`) };
}

function experiment(text) {
  return { outputs: { marked: { text, inspection: {
    fraction: 0.6, control_fraction: 0.5, events: 2, trials: 60,
    series: [{ characters: 1, matching: 0.7, control: 0.5 },
      { characters: [...text].length, matching: 0.6, control: 0.5 }],
  } } } };
}

test('Python code-point prefixes align with chart endpoints and pointer scrubbing', () => {
  const app = ui();
  const text = '🌱 A seed grows. 🌳 A tree lives. 🌍 Life continues.';
  app.set('experiment', experiment(text));
  app.get('scrub').value = '0'; app.run('chart()');
  assert.equal(app.get('prefix').textContent, '🌱');
  app.get('scrub').value = '1'; app.run('chart()');
  assert.equal(app.get('prefix').textContent, text);
  const chart = app.get('chart').children;
  assert.ok(chart.some(n => n.textContent === `${[...text].length} characters`));
  assert.ok(chart.some(n => n.attributes.class === 'cursor' && n.attributes.x1 === '544'));
  app.run('scrubChart({clientX: 544})');
  assert.equal(Number(app.get('scrub').value), 1);
  assert.equal(app.get('prefix').textContent, text);
});

test('Keep first half cannot split a supplementary character without spaces', () => {
  const app = ui();
  // Disable the asynchronous inspect action; test only the actual click transform.
  app.set('busy', true);
  app.get('edited').value = '🌱🌳🌍';
  app.get('half').listeners.click();
  assert.equal(app.get('edited').value, '🌱🌳');
});

test('Undoing pending edits restores the matching metric label for original and edited text', () => {
  const app = ui(); app.set('experiment', experiment('Original.'));
  for (const edited of [false, true]) {
    const text = edited ? 'Measured edit.' : 'Original.';
    app.set('measuredText', edited ? text : null);
    app.set('editMeasurement', edited ? { inspection: { fraction: 0.4, events: 3, trials: 90 } } : null);
    app.get('edited').value = text + ' extra'; app.run('markDirty()');
    assert.match(app.get('measurement-status').textContent, /pending edits/);
    app.get('edited').value = text; app.run('markDirty()');
    assert.equal(app.get('edit-status').textContent, 'Displayed measurements match this text.');
    assert.match(app.get('measurement-status').textContent, edited ? /Last measured edit/ : /Original marked response/);
    assert.equal(app.get('fraction').textContent, edited ? '40.0%' : '60.0%');
  }
});

test('Response switch changes only display selection and accessible pressed state', () => {
  const app = ui(); app.set('experiment', experiment('Keep the actual output.'));
  app.get('read-ordinary').listeners.click();
  assert.equal(app.get('outputs').attributes['data-focus'], 'ordinary');
  assert.equal(app.get('read-ordinary').attributes['aria-pressed'], 'true');
  assert.equal(app.get('read-marked').attributes['aria-pressed'], 'false');
  app.get('read-marked').listeners.click();
  assert.equal(app.get('outputs').attributes['data-focus'], 'marked');
  assert.equal(app.get('read-marked').attributes['aria-pressed'], 'true');
  app.run('readResponse("invalid")');
  assert.equal(app.get('outputs').attributes['data-focus'], 'marked');
  assert.equal(app.run('experiment.outputs.marked.text'), 'Keep the actual output.');
});

test('Prompt disclosure preview follows literal input without generating', () => {
  const app = ui();
  app.get('prompt').value = '  <script>text</script> 🌱  ';
  app.get('prompt').listeners.input();
  assert.equal(app.get('prompt-preview').textContent, '<script>text</script> 🌱');
  app.get('prompt').value = '   ';
  app.get('prompt').listeners.input();
  assert.match(app.get('prompt-preview').textContent, /Enter a prompt/);
  app.run('restoreRequest({last_attempt:{request:{action:"generate",text:"Restored input."}}})');
  assert.equal(app.get('prompt-preview').textContent, 'Restored input.');
  assert.equal(app.run('experiment'), null);
});

test('Global stop stays available during generation with collapsed prompt controls', () => {
  const app = ui();
  app.set('busy', true); app.set('stopAvailable', true); app.set('activeAction', 'generate');
  app.run('updateStopControls()');
  assert.equal(app.get('stop').hidden, false);
  assert.equal(app.get('stop').disabled, false);
  assert.equal(app.get('stop-edit').hidden, true);
  app.set('activeAction', 'inspect'); app.run('updateStopControls()');
  assert.equal(app.get('stop-edit').hidden, false);
  app.set('busy', false); app.run('updateStopControls()');
  assert.equal(app.get('stop').hidden, true);
  assert.equal(app.get('stop-edit').hidden, true);
});


test('Backend labels distinguish GGUF and never invent a model for unknown identities', () => {
  const app = ui();
  assert.equal(app.run('modelLabel({profile: "gguf-byte-bpe-v1-experimental"})'), 'llama.cpp · experimental CPU');
  assert.equal(app.run('modelLabel({profile: "portable-bytelevel-v1-experimental"})'), 'Transformers · experimental CPU');
  assert.equal(app.run('modelLabel({profile: "unknown"})'), 'Local model');
  assert.equal(app.run('modelLabel({})'), 'Local model');
});
