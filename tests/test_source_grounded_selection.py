from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from prepare_source_grounded import select


def records():
    return [{"category": "summarization", "instruction": f"Summarize record {i}",
             "context": f"Record {i}: " + "source text " * 150, "response": f"Reference {i}"} for i in range(20)]


def identities(rows):
    return [(r["source_sha256"], r["instruction_sha256"]) for r in rows]


def test_selection_ignores_reference_answers_and_file_order():
    rows = records()
    chosen, counts = select(rows)
    assert len(chosen) == 16 and counts["eligible_unique_sources"] == 20
    for row in rows:
        row["response"] = "completely different incorrect answer"
    assert identities(select(list(reversed(rows)))[0]) == identities(chosen)


def test_duplicates_and_length_boundaries_are_explicit():
    rows = records()
    rows += [{**rows[0], "instruction": "Another instruction for same source"},
             {**rows[0], "context": "x" * 1199}, {**rows[0], "context": "x" * 6001},
             {**rows[0], "category": "open_qa", "context": "other " * 300}]
    chosen, counts = select(rows)
    assert counts == {"eligible_records": 21, "eligible_unique_sources": 20}
    assert len({r["source_sha256"] for r in chosen}) == 16
    assert identities(chosen) == identities(select(list(reversed(rows)))[0])


def test_insufficient_source_pool_does_not_relax_selection():
    with pytest.raises(ValueError, match="Insufficient"):
        select(records()[:2])
