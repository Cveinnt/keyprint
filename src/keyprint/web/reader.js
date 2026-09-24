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

// Exact-text alignment for reading, not sampler attribution. Bound the dynamic
// programming table; long dissimilar middles fall back to a disclosed block diff.
function wordingDifference(ordinary, marked) {
  const split = text => text.match(/\s+|[\p{L}\p{N}\p{M}_]+|[^\s]/gu) || [];
  const a = split(ordinary), b = split(marked);
  const sharedA = new Uint8Array(a.length), sharedB = new Uint8Array(b.length);
  let start = 0, endA = a.length, endB = b.length;
  while (start < endA && start < endB && a[start] === b[start]) {
    sharedA[start] = sharedB[start] = 1; start++;
  }
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    sharedA[--endA] = sharedB[--endB] = 1;
  }
  const n = endA - start, m = endB - start;
  const bounded = (n + 1) * (m + 1) > 250000;
  if (!bounded && n && m) {
    const width = m + 1, lengths = new Uint32Array((n + 1) * width);
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        lengths[i * width + j] = a[start + i] === b[start + j]
          ? 1 + lengths[(i + 1) * width + j + 1]
          : Math.max(lengths[(i + 1) * width + j], lengths[i * width + j + 1]);
      }
    }
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (a[start + i] === b[start + j]) {
        sharedA[start + i++] = 1; sharedB[start + j++] = 1;
      } else if (lengths[(i + 1) * width + j] >= lengths[i * width + j + 1]) i++;
      else j++;
    }
  }
  function passages(parts, shared) {
    const result = [];
    parts.forEach((text, i) => {
      const different = !shared[i], prior = result[result.length - 1];
      if (prior && prior.different === different) prior.text += text;
      else result.push({text, different});
    });
    return result;
  }
  return {ordinary: passages(a, sharedA), marked: passages(b, sharedB),
    identical: ordinary === marked, bounded};
}

function renderWordingPair(ordinaryTarget, markedTarget, ordinary, marked) {
  const result = wordingDifference(ordinary, marked);
  for (const [condition, target] of [['ordinary', ordinaryTarget], ['marked', markedTarget]]) {
    target.replaceChildren(); target.classList.toggle('reading', false);
    for (const passage of result[condition]) {
      if (passage.different) {
        const node = target.ownerDocument.createElement('mark');
        node.className = 'wording-difference'; node.textContent = passage.text;
        target.append(node);
      } else target.append(target.ownerDocument.createTextNode(passage.text));
    }
  }
  return result.identical
    ? 'Identical wording in these two samples. Nothing is hidden or substituted.'
    : (result.bounded ? 'Long passages are compared as blocks. ' : '') +
      'Highlights show wording differences between independent samples, not which words are watermarked. Exact text is preserved.';
}

if (typeof module !== "undefined") module.exports = { renderResponse, wordingDifference, renderWordingPair };
