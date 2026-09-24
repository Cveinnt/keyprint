const {test} = require('node:test');
const assert = require('node:assert/strict');
const {wordingDifference, renderWordingPair} = require('../src/keyprint/web/reader.js');

function assertExact(a, b) {
  const result = wordingDifference(a, b);
  assert.equal(result.ordinary.map(p => p.text).join(''), a);
  assert.equal(result.marked.map(p => p.text).join(''), b);
  assert.equal(result.ordinary.filter(p => !p.different).map(p => p.text).join(''),
               result.marked.filter(p => !p.different).map(p => p.text).join(''));
  return result;
}

test('insertions, deletions and replacements retain shared context', () => {
  const r = assertExact('Maya must review by Tuesday.', 'Maya must approve by Friday.');
  assert.deepEqual(r.ordinary.filter(p => p.different).map(p => p.text), ['review', 'Tuesday']);
  assert.deepEqual(r.marked.filter(p => p.different).map(p => p.text), ['approve', 'Friday']);
  assertExact('Keep this.', 'Keep only this.');
  assertExact('Keep only this.', 'Keep this.');
  assertExact('A B A B', 'B A B A');
});

test('identical and empty samples remain explicit without invented differences', () => {
  for (const text of ['', 'Same text', 'Français 中文 👩🏽‍💻\r\n\t']) {
    const r = assertExact(text, text);
    assert.equal(r.identical, true);
    assert.ok(r.ordinary.every(p => !p.different));
  }
  assertExact('', 'Text'); assertExact('Text', '');
});

test('Unicode, combining marks and original whitespace survive alignment', () => {
  assertExact('Été e\u0301 中文 👩🏽‍💻\r\n\t  A', 'Été é 日本語 👩🏽‍💻\n B');
});

test('large dissimilar middles use bounded comparison without truncation', () => {
  const a = 'start ' + 'a '.repeat(1500) + 'end';
  const b = 'start ' + 'b '.repeat(1500) + 'end';
  const r = assertExact(a, b);
  assert.equal(r.bounded, true);
  assert.equal(r.ordinary[0].text, 'start ');
  assert.equal(r.ordinary.at(-1).text, ' end');
});

test('rendered differences keep HTML inert and source characters exact', () => {
  class Node {
    constructor(tag) { assert.ok(['div', 'mark', '#text'].includes(tag)); this.tag = tag; this.children = []; this.ownerDocument = doc; this.classList = {toggle(){}}; }
    set innerHTML(_) { throw new Error('Forbidden HTML sink'); }
    replaceChildren() { this.children = []; this.text = undefined; }
    append(node) { this.children.push(node); }
    set textContent(text) { this.text = text; this.children = []; }
    get textContent() { return this.text ?? this.children.map(n => n.textContent).join(''); }
  }
  const doc = {createElement: tag => new Node(tag), createTextNode: text => { const n = new Node('#text'); n.textContent = text; return n; }};
  const a = doc.createElement('div'), b = doc.createElement('div');
  const source = '<img src=x onerror=alert(1)>\r\n';
  const note = renderWordingPair(a, b, 'plain', source);
  assert.equal(a.textContent, 'plain'); assert.equal(b.textContent, source);
  assert.match(note, /independent samples, not which words are watermarked/);
  assert.match(renderWordingPair(a, b, source, source), /Identical wording/);
  assert.equal(b.textContent, source);
});
