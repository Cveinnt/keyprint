import json

import pytest

torch = pytest.importorskip("torch")

from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental.native import RequestSampler


def binding():
    data = {"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1, "B": 2, "C": 3}},
            "decoder": {"type": "ByteLevel"}, "normalizer": None,
            "added_tokens": [{"id": 0, "content": "<eos>", "special": True}]}
    return ByteLevelBinding.create(json.dumps(data), vocabulary_size=4, special_ids=[0], eos_ids=[0])


def test_selected_path_must_match_host_before_next_draw(tmp_path):
    sampler = RequestSampler(binding(), bytes(range(32)), "marked", tmp_path, runtime="keyprint")
    try:
        first = sampler([], torch.tensor([-float("inf"), 1., 2., 3.])).argmax().item()
        assert sampler.selected == [first]
        with pytest.raises(RuntimeError, match="host prefix"):
            sampler([], torch.tensor([-float("inf"), 1., 2., 3.]))
        assert len(sampler.selected) == 1
        second = sampler([first], torch.tensor([-float("inf"), 3., 2., 1.])).argmax().item()
        assert sampler.selected == [first, second]
    finally:
        sampler.close()
    rows = [json.loads(line)["event"] for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert len([r for r in rows if r.get("phase") == "selected_tentative"]) == 2
    assert rows[0]["scope"].startswith("experimental selections")


@pytest.mark.parametrize("head", [torch.tensor([1., 2.]), torch.tensor([0., 1., 2., float("nan")]),
    torch.tensor([0., 1., 2., float("inf")]), torch.tensor([0., 1., 2., 3.], dtype=torch.float64)])
def test_invalid_head_fails_before_selection(tmp_path, head):
    sampler = RequestSampler(binding(), bytes(range(32)), "ordinary", tmp_path, runtime="keyprint")
    try:
        with pytest.raises(ValueError):
            sampler([], head)
        assert sampler.selected == []
    finally:
        sampler.close()
