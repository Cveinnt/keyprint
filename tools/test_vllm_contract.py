"""Run in the pinned vLLM CPU image; exercises upstream batch-state operations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, BatchUpdate, MoveDirectionality

from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental.vllm import KeyprintLogitsProcessor, RequestSampler, validate


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        data = {"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1, "B": 2, "C": 3}},
                "decoder": {"type": "ByteLevel"}, "normalizer": None,
                "added_tokens": [{"id": 0, "content": "<eos>", "special": True}]}
        self.binding = ByteLevelBinding.create(json.dumps(data), vocabulary_size=4, special_ids=[0], eos_ids=[0])
        self.adapter = object.__new__(KeyprintLogitsProcessor)
        AdapterLogitsProcessor.__init__(self.adapter, None, None, None)
        self.adapter.binding, self.adapter.key, self.adapter.directory = self.binding, bytes(range(32)), self.directory
        self.params = SamplingParams(max_tokens=4, temperature=1., top_p=1., top_k=-1)

    def tearDown(self):
        for partial in self.adapter.req_info.values():
            partial.func.close()
        self.temp.cleanup()

    def test_swap_remove_replace_and_prefix_isolation(self):
        first, second = [], []
        self.adapter.update_state(BatchUpdate(2, [], [(0, self.params, [1], first), (1, self.params, [2], second)], []))
        a, b = self.adapter.req_info[0].func, self.adapter.req_info[1].func
        selected = self.adapter.apply(torch.tensor([[-float("inf"), 1., 2., 3.]] * 2)).argmax(-1).tolist()
        first.append(selected[0]); second.append(selected[1])
        self.adapter.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
        self.assertIs(self.adapter.req_info[0].func, b)
        self.assertIs(self.adapter.req_info[1].func, a)
        selected = self.adapter.apply(torch.tensor([[-float("inf"), 3., 1., 2.]] * 2)).argmax(-1).tolist()
        second.append(selected[0]); first.append(selected[1])
        self.assertEqual(first, a.selected)
        self.assertEqual(second, b.selected)
        with patch.object(b, "close", wraps=b.close) as closed:
            self.adapter.update_state(BatchUpdate(1, [0], [], [(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
            closed.assert_called_once()
        self.assertIs(self.adapter.req_info[0].func, a)
        with patch.object(a, "close", wraps=a.close) as closed:
            self.adapter.update_state(BatchUpdate(1, [], [(0, self.params, [3], [])], []))
            closed.assert_called_once()
        self.assertEqual(self.adapter.req_info[0].func.selected, [])

    def test_mismatched_host_prefix_fails_without_second_draw(self):
        sampler = RequestSampler(self.binding, bytes(range(32)), "marked", self.directory)
        try:
            sampler([], torch.tensor([-float("inf"), 1., 2., 3.]))
            with self.assertRaisesRegex(RuntimeError, "host prefix"):
                sampler([], torch.tensor([-float("inf"), 1., 2., 3.]))
            self.assertEqual(len(sampler.selected), 1)
        finally:
            sampler.close()

    def test_unsupported_sampling_controls_rejected(self):
        for setting in ({"temperature": .7}, {"top_k": 5}, {"seed": 7}, {"n": 2},
                        {"logprobs": 1}, {"stop": ["end"]}, {"ignore_eos": True},
                        {"extra_args": {"keyprint_condition": "other"}}):
            values = dict(max_tokens=4, temperature=1., top_p=1., top_k=-1)
            values.update(setting)
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                validate(SamplingParams(**values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
