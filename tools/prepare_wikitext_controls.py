"""Prepare pinned, article-grouped controls without looking at detector scores.

Run in a separate data environment with pyarrow==25.0.1. Text stays outside
public/. Preserve source punctuation/spacing, including WikiText @-@ artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
FILES = {
    "train-00000-of-00002.parquet": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
    "train-00001-of-00002.parquet": "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720",
}
CARD = f"https://huggingface.co/datasets/Salesforce/wikitext/blob/{REVISION}/README.md"
SALT = "keyprint-wikitext-length-controls-v1"
LENGTHS = (300, 600)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prefix(text, count):
    for i, match in enumerate(re.finditer(r"\S+", text), 1):
        if i == count:
            return text[:match.end()]
    return None


def articles(rows):
    """One article across consecutive shards; section headings remain body text."""
    title, start, body = None, None, []
    index = -1
    for index, text in enumerate(rows):
        heading = re.fullmatch(r"=\s+([^=].*?)\s+=", text.strip())
        if heading:
            if title is not None:
                yield {"title": title, "start_row": start, "end_row": index - 1,
                       "body": "".join(body)}
            title, start, body = heading.group(1), index, []
        elif title is not None:
            body.append(text)
        elif text.strip():
            raise ValueError("Nonempty source text precedes the first article")
    if title is not None:
        yield {"title": title, "start_row": start, "end_row": index,
               "body": "".join(body)}


def select(rows, per_length):
    if type(per_length) is not int or per_length < 1:
        raise ValueError("Positive per-length count required")
    eligible, total = [], 0
    for article in articles(rows):
        total += 1
        text = prefix(article.pop("body"), max(LENGTHS))
        if text is not None:
            eligible.append({**article, "text": text})
    # Transitive grouping prevents a shared title or opening from entering both
    # length strata. Longer and shorter passages come from the same eligible pool.
    parent = list(range(len(eligible)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, article in enumerate(eligible):
        for name, value in (("title", article["title"]),
                            ("opening", prefix(article["text"], min(LENGTHS)))):
            fingerprint = (name, sha(normalized(value).encode()))
            if fingerprint in seen:
                parent[root(i)] = root(seen[fingerprint])
            seen[fingerprint] = i
    groups = {}
    for i, article in enumerate(eligible):
        groups.setdefault(root(i), []).append(article)
    representatives = [min(group, key=lambda a: a["start_row"]) for group in groups.values()]
    representatives.sort(key=lambda a: sha((SALT + normalized(a["title"])).encode()))
    needed = per_length * len(LENGTHS)
    if len(representatives) < needed:
        raise ValueError(f"Need {needed} exact groups; found {len(representatives)}")
    selected = []
    for i, article in enumerate(representatives[:needed]):
        words = LENGTHS[i % len(LENGTHS)]
        text = prefix(article["text"], words)
        selected.append({**article, "words": words, "text": text,
                         "text_sha256": sha(text.encode()),
                         "opening_sha256": sha(normalized(prefix(text, min(LENGTHS))).encode())})
    return selected, {"source_articles": total, "eligible_at_least_600_words": len(eligible),
                      "exact_groups": len(groups), "per_length": per_length,
                      "planned_documents": needed,
                      "limitation": "Exact normalized titles/openings only; authors, topics and near duplicates may remain dependent"}


def parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            yield from batch.column(0).to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--per-length", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.source_dir / name for name in FILES]
    for path in paths:
        if sha(path.read_bytes()) != FILES[path.name]:
            raise ValueError(f"Pinned source checksum differs: {path.name}")
    selected, selection = select(parquet_rows(paths), args.per_length)
    args.output.mkdir(mode=0o700)
    public = args.output / "public"
    public.mkdir()
    with (args.output / "texts.jsonl").open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {"source": "Salesforce/wikitext", "revision": REVISION,
                "config": "wikitext-103-raw-v1", "split": "train", "source_files": FILES,
                "card_url": CARD,
                "license_note": "Pinned card metadata lists CC BY-SA 3.0 and GFDL; prose lists CC BY-SA 4.0. Preserve this discrepancy; do not relicense or export source text.",
                "attribution": "Wikipedia contributors; WikiText curated by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher (2016)",
                "selection": selection, "salt": SALT, "lengths": list(LENGTHS),
                "rule": "One article per exact group; hash-ranked titles; alternate 300/600 whitespace-word prefixes from articles with at least 600 words",
                "text_processing": "Exclude top-level title; concatenate original body rows and cut at word boundary. Preserve section headings, whitespace and tokenized punctuation; no detokenization.",
                "script_sha256": sha(Path(__file__).read_bytes()),
                "texts_sha256": sha((args.output / "texts.jsonl").read_bytes()),
                "records": [{k: v for k, v in row.items() if k != "text"} for row in selected]}
    write(public / "manifest.json", manifest)
    (public / "ATTRIBUTION.md").write_text(
        f"# WikiText controls\n\n{manifest['attribution']}.\n\n[Source card]({CARD}).\n\n"
        f"{manifest['license_note']}\n\n{manifest['text_processing']}\n\n"
        "Only article metadata, hashes and derived scores belong in public/.\n", encoding="utf-8")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
