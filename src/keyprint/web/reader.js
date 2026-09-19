"use strict";

// A deliberately small reading view, not a Markdown/HTML interpreter. Untrusted
// output only enters text nodes; links, images and HTML are never activated.
// The caller retains the exact source for inspection, export and source view.
function renderResponse(target, source, exact = false) {
  target.replaceChildren();
  target.classList.toggle("reading", !exact);
  if (exact || !source) {
    target.textContent = source || "(Empty model response)";
    return;
  }
  const doc = target.ownerDocument;
  function node(tag, text) {
    const result = doc.createElement(tag);
    if (text !== undefined) result.textContent = text;
    return result;
  }
  function inline(parent, text) {
    // Unmatched or nested syntax remains literal. Code takes precedence.
    const pattern = /`([^`\n]+)`|\*\*([^*`\n]+)\*\*/g;
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(doc.createTextNode(text.slice(cursor, match.index)));
      parent.append(node(match[1] === undefined ? "strong" : "code", match[1] ?? match[2]));
      cursor = match.index + match[0].length;
    }
    parent.append(doc.createTextNode(text.slice(cursor)));
  }
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  let paragraph = [], list = null;
  function flush() {
    if (!paragraph.length) return;
    const p = node("p");
    inline(p, paragraph.join("\n"));
    target.append(p);
    paragraph = [];
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s{0,3}(`{3,}|~{3,})[^`~]*$/);
    if (fence) {
      flush(); list = null;
      const code = [];
      const closing = new RegExp("^\\s{0,3}" + fence[1][0] + "{" + fence[1].length + ",}\\s*$");
      while (++i < lines.length && !closing.test(lines[i])) code.push(lines[i]);
      const pre = node("pre");
      pre.append(node("code", code.join("\n")));
      target.append(pre);
      continue;
    }
    if (!line.trim()) { flush(); list = null; continue; }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const item = line.match(/^(?:([-+*])|(\d{1,9})[.)])\s+(.+)$/);
    if (heading) {
      flush(); list = null;
      const h = node("h4"); inline(h, heading[1]); target.append(h);
    } else if (item) {
      flush();
      const tag = item[2] ? "ol" : "ul";
      if (!list || list.localName !== tag || (tag === "ol" && Number(item[2]) !== list.nextNumber)) {
        list = node(tag);
        if (tag === "ol") list.setAttribute("start", item[2]);
        target.append(list);
      }
      const li = node("li"); inline(li, item[3]); list.append(li);
      if (tag === "ol") list.nextNumber = Number(item[2]) + 1;
    } else {
      list = null;
      paragraph.push(line);
    }
  }
  flush();
}

if (typeof module !== "undefined") module.exports = { renderResponse };
