import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest


def load(name, monkeypatch):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def body(label):
    return " " + " ".join(f"{label}-{i}" for i in range(620)) + "\n"


def test_article_boundary_crosses_shards_and_retains_sections(monkeypatch):
    tool = load("prepare_wikitext_controls", monkeypatch)
    shards = [["", " = A = \n", " first @-@ paragraph\n"],
              [" = = Section = = \n", " second paragraph\n", " = B = \n", " last\n"]]
    result = list(tool.articles(itertools.chain.from_iterable(shards)))
    assert len(result) == 2
    assert result[0] == {"title": "A", "start_row": 1, "end_row": 4,
                         "body": " first @-@ paragraph\n = = Section = = \n second paragraph\n"}
    assert result[1]["body"] == " last\n"
    with pytest.raises(ValueError, match="precedes"):
        list(tool.articles(["missing heading"]))


def test_transitive_groups_never_cross_length_strata(monkeypatch):
    tool = load("prepare_wikitext_controls", monkeypatch)
    # A/B share title; B/C share their 300-word opening: one group, not two.
    rows = [" = Same = \n", body("a"), " = same = \n", body("b"),
            " = Other = \n", body("b")]
    for i in range(7):
        rows.extend([f" = Unique {i} = \n", body(str(i))])
    rows.extend([" = Too short = \n", " brief text\n"])
    result, stats = tool.select(iter(rows), 4)
    assert stats["source_articles"] == 11
    assert stats["eligible_at_least_600_words"] == 10
    assert stats["exact_groups"] == 8
    assert len({r["opening_sha256"] for r in result}) == 8
    assert [r["words"] for r in result] == [300, 600] * 4
    for row in result:
        assert len(row["text"].split()) == row["words"]
        assert row["text"].startswith(" ")  # spacing is source data
        assert row["text_sha256"] == tool.sha(row["text"].encode())
    assert result == tool.select(iter(rows), 4)[0]
    with pytest.raises(ValueError, match="found 8"):
        tool.select(iter(rows), 5)


def test_manifest_rejects_changed_text_and_counts_missing_scores(monkeypatch, tmp_path):
    for name in ("compare_score_baselines", "validate_null_corpus", "weighted_null", "layer_likelihood", "prepare_wikitext_controls"):
        load(name, monkeypatch)
    tool = load("validate_wikitext_null", monkeypatch)
    prep = sys.modules["prepare_wikitext_controls"]
    records, selection = prep.select(iter([" = First = \n", body("a"), " = Second = \n", body("b")]), 1)
    raw = "".join(json.dumps(row) + "\n" for row in records).encode()
    (tmp_path / "public").mkdir()
    manifest = {"revision": tool.REVISION, "source_files": tool.FILES,
                "texts_sha256": tool.sha(raw), "selection": selection,
                "records": [{k: v for k, v in row.items() if k != "text"} for row in records]}
    (tmp_path / "texts.jsonl").write_bytes(raw)
    (tmp_path / "public/manifest.json").write_text(json.dumps(manifest))
    assert tool.load_controls(tmp_path)[1] == records
    (tmp_path / "texts.jsonl").write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        tool.load_controls(tmp_path)
    summary = tool.panel([{"flagged": True}, {"error": "ValueError"}], 2)
    assert summary == {"planned": 2, "available": 1, "false_hits": 1,
                       "unavailable": 1, "iid_only_upper_97_5_percent": None}
