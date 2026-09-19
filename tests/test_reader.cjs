// No browser dependency: assert the renderer's DOM contract with a strict sink.
// Actual rendering, keyboard and refresh flows also need browser QA.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { renderResponse } = require("../src/keyprint/web/reader.js");
const allowed = new Set(["div", "p", "strong", "code", "h4", "ul", "ol", "li", "pre", "#text"]);
class Node {
  constructor(tag) {
    assert.ok(allowed.has(tag), `Unexpected element ${tag}`);
    this.localName = tag; this.children = []; this.attributes = {};
    this.ownerDocument = doc; this.classList = { toggle() {} };
  }
  set innerHTML(_) { throw new Error("HTML sink is forbidden"); }
  set textContent(text) { this.children = []; this.text = text; }
  get textContent() { return this.text ?? this.children.map(n => n.textContent).join(""); }
  replaceChildren() { this.children = []; this.text = undefined; }
  append(...children) { this.children.push(...children); }
  setAttribute(k, v) { assert.equal(k, "start"); this.attributes[k] = v; }
}
const doc = {
  createElement: tag => new Node(tag),
  createTextNode: text => { const n = new Node("#text"); n.textContent = text; return n; },
};
test("model HTML, links, images and script syntax stay inert", () => {
  const root = doc.createElement("div");
  const text = '<img src=x onerror=alert(1)>\n<script>alert(2)</script>\n[click](javascript:alert(3))\n![image](https://example.com/track)';
  renderResponse(root, text);
  assert.equal(root.textContent, text);
  assert.equal(root.children[0].localName, "p");
});
test("headings, emphasis, list numbering and fenced code read correctly", () => {
  const root = doc.createElement("div");
  renderResponse(root, '# A **guide**\n\n3. First\n4. Second\n9. Ninth\n\n```html\n<img src=x>\n**literal**\n```');
  assert.deepEqual(root.children.map(n => n.localName), ["h4", "ol", "ol", "pre"]);
  assert.equal(root.children[0].textContent, "A guide");
  assert.equal(root.children[0].children[1].localName, "strong");
  assert.equal(root.children[1].attributes.start, "3");
  assert.equal(root.children[1].children.length, 2);
  assert.equal(root.children[2].attributes.start, "9");
  assert.equal(root.children[3].textContent, '<img src=x>\n**literal**');
});
test("exact mode round-trips whitespace, Unicode and unfinished Markdown", () => {
  const root = doc.createElement("div");
  const text = '  Français 日本語 🙂\r\n\t**unfinished\n```py\nx = 2\n';
  renderResponse(root, text);
  assert.ok(root.textContent.includes('x = 2'));
  renderResponse(root, text, true);
  assert.equal(root.textContent, text);
  renderResponse(root, '**new**');
  assert.equal(root.textContent, 'new');
});
test("unmatched inline markers stay visible and code does not parse emphasis", () => {
  const root = doc.createElement("div");
  renderResponse(root, 'Keep **unfinished and `**literal**` as code.');
  assert.equal(root.textContent, 'Keep **unfinished and **literal** as code.');
  assert.equal(root.children[0].children[1].localName, "code");
});
