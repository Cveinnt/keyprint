"use strict";

// Share a recorded example, never a prompt, session token or private URL query.
function recordingShareURL(href, index) {
  const url = new URL(href);
  if (!['https:', 'http:'].includes(url.protocol) ||
      !Number.isInteger(index) || index < 0 || index > 11)
    throw new Error('Choose a recorded example before sharing.');
  url.username = '';
  url.password = '';
  url.search = '';
  url.hash = 'gallery';
  url.searchParams.set('example', String(index));
  return url.href;
}

if (typeof document !== 'undefined') {
  document.getElementById('share-example')?.addEventListener('click', async () => {
    const buttons = Array.from(document.querySelectorAll('#gallery-cards button'));
    const index = buttons.findIndex(button => button.getAttribute('aria-pressed') === 'true');
    const status = document.getElementById('share-status');
    try {
      const link = recordingShareURL(location.href, index);
      const field = document.getElementById('share-link');
      field.value = link;
      field.hidden = false;
      try {
        await navigator.clipboard.writeText(link);
        status.textContent = 'Link copied. It includes only the recorded example number.';
      } catch {
        field.focus();
        field.select();
        status.textContent = 'Select and copy the link above. No prompt or private session data is included.';
      }
      if (['localhost', '127.0.0.1', '[::1]'].includes(location.hostname))
        status.textContent += ' This is a local link; publish the downloaded viewer for others to use.';
    } catch (error) {
      status.textContent = error.message;
    }
  });
}
if (typeof module !== 'undefined') module.exports = {recordingShareURL};
