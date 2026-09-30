"use strict";

function mountGallery(doc, data, onSelect, initialIndex = 0) {
  if (data?.schema !== 'keyprint-gallery-v1' || !Array.isArray(data.examples) ||
      data.examples.length < 1 || data.examples.length > 12 ||
      data.examples.some(e => typeof e.title !== 'string' || !e.title.trim() ||
        e.title.length > 80 || (e.note != null && (typeof e.note !== 'string' || e.note.length > 1000)) || e.recording?.schema !== 'keyprint-comparison-v1' ||
        typeof e.recording.prompt !== 'string' ||
        !['ordinary','marked'].every(c => typeof e.recording.experiment?.outputs?.[c]?.text === 'string')))
    throw new Error('Unsupported gallery format.');
  const container = doc.getElementById('gallery-cards');
  const buttons = [];
  container.replaceChildren();
  function select(index) {
    const example = data.examples[index];
    if (!example) return;
    onSelect(example.recording);
    buttons.forEach((button,i) => button.setAttribute('aria-pressed',String(i === index)));
    doc.getElementById('gallery-current').textContent = `Showing ${example.title}. Recorded output; no model request.`;
    doc.getElementById('gallery-note').textContent = example.note || '';
    doc.getElementById('gallery-note').hidden = !example.note;
  }
  data.examples.forEach((example,index) => {
    const button = doc.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `Open example: ${example.title}`);
    const number = doc.createElement('span');
    number.className = 'eyebrow';
    number.textContent = String(index + 1).padStart(2,'0');
    const title = doc.createElement('strong'); title.textContent = example.title;
    const excerpt = doc.createElement('span'); excerpt.className = 'gallery-excerpt';
    const chars = Array.from(example.recording.experiment.outputs.marked.text);
    excerpt.textContent = chars.slice(0,110).join('') + (chars.length > 110 ? '…' : '');
    button.append(number,title,excerpt);
    button.addEventListener('click',()=>select(index));
    buttons.push(button);container.append(button);
  });
  doc.getElementById('gallery').hidden = false;
  select(Number.isInteger(initialIndex) && initialIndex >= 0 && initialIndex < data.examples.length ? initialIndex : 0);
  return {select};
}
if (typeof module !== 'undefined') module.exports = {mountGallery};
