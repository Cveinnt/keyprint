from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from replay_wide_mlx import ordinary_weights, replay_draw
from summarize_wide_fidelity import summary
from test_long_fidelity import fixture
from keyprint.experimental.wide_sampling import support_filter, softmax, sample


@pytest.mark.parametrize("temperature", [.7, 1., 1e-35, 1e308])
def test_independent_gap_first_model_head_matches_production(temperature):
    rng = np.random.default_rng(374)
    raw = rng.uniform(-1000, 2, (1, 248320)).astype(np.float32)
    raw[0, [0, 200000, 248319]] = 10
    pieces = [b"word"] * 248320
    pieces[0] = pieces[248319] = None
    binding = SimpleNamespace(pieces=pieces, eos_ids={248319})
    expected_raw = raw.copy()
    expected_raw[0, 0] = -np.inf
    expected = softmax(support_filter(expected_raw, temperature=temperature, top_k=100).filtered_logits[0])
    actual = ordinary_weights(raw, binding, temperature=temperature, top_k=100)
    assert actual.tobytes() == expected.tobytes()


def test_recorded_draw_is_replayed_without_new_randomness_and_detects_change():
    weights = np.zeros(248320)
    weights[[4, 200000, 248319]] = [.2, .3, .5]
    expected = json.loads(json.dumps(asdict(sample(weights, random.Random(1).getrandbits))))
    assert replay_draw(weights, expected) == expected["token_index"]
    expected["token_index"] = 100
    with pytest.raises(ValueError, match="selection/transcript differs"):
        replay_draw(weights, expected)


def test_scalar_reference_source_weights_match_sparse_profile():
    from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
    from keyprint._engine.legacy._impl.research.token_source_policy import TokenSourceSession
    from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
    pieces = [None] * 248320
    for token, word in zip([4, 200000, 248319], [b"word", b" text", b"more"]):
        pieces[token] = word
    profile = Profile(pieces, tokenizer_identity="high-id-fixture", config=Config(max_steps=8))
    reference = TokenSourceSession(profile, bytes(32), condition="marked")
    sparse = SparseTokenSourceSession(profile, bytes(32), condition="marked")
    try:
        weights = np.zeros(248320)
        weights[[4, 200000, 248319]] = [.2, .3, .5]
        for token in [4, 248319, 200000, 4, 248319, 200000]:
            a, b = reference.prepare(weights), sparse.prepare(weights)
            assert a.probabilities.tobytes() == b.probabilities.tobytes()
            reference.commit(a, token); sparse.commit(b, token)
        assert reference.source_receipt() == sparse.source_receipt()
    finally:
        reference.close(); sparse.close()


def test_new_profile_summary_never_reports_missing_detection_as_zero():
    plan, rows, ratings = fixture()
    for row in rows:
        row.pop("matching_hit"); row.pop("other_hit")
    result = summary(plan, rows, ratings)
    assert result["groups"]["marked"]["task_pass"] == 1
    assert "matching_hits" not in result["groups"]["marked"]
    assert "joint_pass" not in result["judgments"]["marked"]
    assert "joint_pass" not in result["paired"]
    assert result["detector_calibrated"] is False
    ratings[1]["fact_checks"] = [False]
    assert summary(plan, rows, ratings)["groups"]["marked"]["content_pass"] == 0
