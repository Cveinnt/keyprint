"""Build or serve the public recorded gallery. Standard library; no inference."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "app.js", "app.css", "gallery.js", "trace.js", "reader.js", "favicon.svg"):
        shutil.copy2(ROOT / "src/keyprint/web" / name, destination / name)
    for name in ("gallery.json", "replay.json", "provenance.json"):
        shutil.copy2(ROOT / "demos/recordings" / name, destination / name)
    page = destination / "index.html"
    html = page.read_text().replace('<body>', '<body data-mode="replay" data-gallery="true">')
    html = html.replace('<body data-mode="live">', '<body data-mode="replay" data-gallery="true">')
    if 'data-mode="replay"' not in html:
        raise ValueError("Missing static replay mode")
    html = html.replace('Keyprint — the local playground', 'Keyprint · Play with real generations')
    html = html.replace('<main>', '<a class="skip-demo" href="#gallery">Skip to the demos</a><main>')
    html = html.replace('<a href="#top" class="wordmark">', '<a href="https://keyprint.vercel.app/" class="wordmark">')
    html = html.replace('The local playground · community research preview', 'Real SDK recordings · no install or account')
    html = html.replace('<a href="#method">How to read this</a>', '<a href="https://github.com/Cveinnt/keyprint">Code &amp; remix</a>')
    html = html.replace('<h1>Ordinary words. A pattern underneath.</h1>', '<h1>One prompt.<br>Two paths through language.</h1>')
    html = html.replace('Try your own prompt ↗', 'Make your own · edit a prompt and get Python')
    html = html.replace('<section id="gallery"', '<nav class="demo-links" aria-label="Remix this demo"><a href="./keyprint-demo.zip" download>Download this demo</a><a href="./gallery.json">Get the data</a><a href="https://github.com/Cveinnt/keyprint/tree/community-preview/demos">Remix guide</a><a href="https://github.com/Cveinnt/keyprint/discussions/categories/show-and-tell">Share yours</a></nav><section id="gallery"')
    html = html.replace('An experiment in text watermarking', 'Generate · compare · remix')
    page.write_text(html)
    with (destination / 'app.css').open('a') as css:
        css.write('\n.demo-links{display:flex;flex-wrap:wrap;gap:12px 26px;padding:22px 0;border-bottom:1px solid var(--line);font:13px ui-monospace,monospace}.demo-links a{text-decoration:underline;text-underline-offset:5px}.skip-demo{position:absolute;left:12px;top:-100px;background:var(--paper);padding:12px;z-index:10}.skip-demo:focus{top:12px}.intro{padding-top:48px;padding-bottom:36px}.intro h1{font-size:clamp(42px,7vw,86px)}\n')
    (destination / 'README.txt').write_text('Keyprint recorded demo. Run: python -m http.server\nOpen http://localhost:8000. No install, account, model weights or API key.\nOriginal public outputs retained, including failures. Signals are uncalibrated diagnostics.\nSource and license: https://github.com/Cveinnt/keyprint\n')
    shutil.copy2(ROOT / 'LICENSE', destination / 'LICENSE')
    # A self-contained shareable viewer; no private inputs, keys or journals.
    with ZipFile(destination / 'keyprint-demo.zip', 'w', ZIP_DEFLATED) as archive:
        for path in sorted(destination.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Build into a new directory and exit')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.output:
        build(args.output)
        print(f'Built {args.output}; no model was loaded.')
        return
    with tempfile.TemporaryDirectory(prefix='keyprint-demo-') as tmp:
        destination = Path(tmp) / 'play'
        build(destination)
        handler = partial(SimpleHTTPRequestHandler, directory=str(destination))
        with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
            print(f'Keyprint demo: http://127.0.0.1:{args.port}/  (Ctrl-C to stop)', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
