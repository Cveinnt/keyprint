const {test} = require('node:test');
const assert = require('node:assert/strict');
const {recordingShareURL} = require('../src/keyprint/web/share.js');

test('recording shares preserve the example but strip private URL fields', () => {
  const url = recordingShareURL('https://user:secret@example.com/play/?prompt=private&token=secret#session=secret', 2);
  assert.equal(url, 'https://example.com/play/?example=2#gallery');
});
test('local and self-hosted remixes keep their own origin and path', () => {
  assert.equal(recordingShareURL('http://127.0.0.1:8000/my-demo/?example=1', 0),
    'http://127.0.0.1:8000/my-demo/?example=0#gallery');
});
test('invalid selection and non-web protocols cannot produce a share URL', () => {
  for (const index of [-1, 12, 1.5, NaN])
    assert.throws(() => recordingShareURL('https://example.com/', index));
  assert.throws(() => recordingShareURL('javascript:alert(1)', 0));
  assert.throws(() => recordingShareURL('file:///private/demo.html', 0));
});
