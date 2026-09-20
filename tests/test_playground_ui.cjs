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
