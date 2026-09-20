"""SGLang CPU contract tests, including processor recreation across batches."""
import gc
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest

import torch
from sglang.srt.sampling.sampling_params import SamplingParams
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental.sglang import KeyprintLogitsProcessor, _LIVE_REQUESTS, _RETIRED_REQUESTS, _reap_finished, validate
from keyprint.experimental.sglang_runtime import verify_runtime, REVISION


class Request:
    def __init__(self):
        self.output_ids = []
        self.return_logprob = False
        self.finished_reason = None
        self.sampling_params = SamplingParams(max_new_tokens=4, temperature=1., top_p=1., top_k=-1)

    def finished(self):
        return self.finished_reason is not None


class Contracts(unittest.TestCase):
    def tearDown(self):
        for state in list(_LIVE_REQUESTS.values()):
            state[-2]()
        _LIVE_REQUESTS.clear()
        _RETIRED_REQUESTS.clear()

    def test_installed_runtime_source_matches_pinned_contract(self):
        distribution = importlib.metadata.distribution('sglang-cpu')
        identity = verify_runtime(distribution.version, Path(distribution.locate_file('sglang')))
        self.assertEqual(identity['source_revision'], REVISION)

    def test_recreated_processor_keeps_request_state_and_rejects_key_change(self):
        data = {"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1, "B": 2, "C": 3}},
                "decoder": {"type": "ByteLevel"}, "normalizer": None,
                "added_tokens": [{"id": 0, "content": "<eos>", "special": True}]}
        with tempfile.TemporaryDirectory() as directory:
            processors = [object.__new__(KeyprintLogitsProcessor) for _ in range(2)]
            for processor in processors:
                processor.binding = ByteLevelBinding.create(json.dumps(data), vocabulary_size=4, special_ids=[0], eos_ids=[0])
                processor.key = bytes(range(32))
                processor.directory = Path(directory)
            req = Request()
            first = processors[0](torch.tensor([[-float("inf"), 1., 2., 3.]]), [{"__req__": req}]).argmax().item()
            req.output_ids.append(first)
            state = _LIVE_REQUESTS[req][-1]
            second = processors[1](torch.tensor([[-float("inf"), 3., 2., 1.]]), [{"__req__": req}]).argmax().item()
            req.output_ids.append(second)
            self.assertIs(_LIVE_REQUESTS[req][-1], state)
            self.assertEqual(state.selected, req.output_ids)
            processors[1].key = bytes(reversed(range(32)))
            with self.assertRaisesRegex(ValueError, "binding, key or condition"):
                processors[1](torch.zeros((1, 4)), [{"__req__": req}])
            self.assertEqual(len(state.selected), 2)
            del req
            gc.collect()
            self.assertTrue(state.journal._file.closed)
            self.assertEqual(len(_LIVE_REQUESTS), 0)

    def test_unsupported_settings_rejected(self):
        for settings in ({"temperature": .7}, {"top_k": 4}, {"ignore_eos": True}, {"sampling_seed": 1},
                         {"stop": ["end"]}, {"json_schema": "{}"}, {"n": 2}):
            args = dict(max_new_tokens=4, temperature=1., top_p=1., top_k=-1)
            args.update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate(SamplingParams(**args))

    def test_finished_cycles_release_slots_without_gc_and_cannot_restart(self):
        data = {"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1}},
                "decoder": {"type": "ByteLevel"}, "normalizer": None,
                "added_tokens": [{"id": 0, "content": "<eos>", "special": True}]}
        with tempfile.TemporaryDirectory() as directory:
            processor = object.__new__(KeyprintLogitsProcessor)
            processor.binding = ByteLevelBinding.create(json.dumps(data), vocabulary_size=2,
                                                        special_ids=[0], eos_ids=[0])
            processor.key = bytes(range(32))
            processor.directory = Path(directory)
            # Strong references and host-style cycles deliberately prevent GC.
            retained = []
            for i in range(72):
                req = Request()
                req.sampling_params.custom_params = {"__req__": req}
                processor(torch.tensor([[-float('inf'), 0.]]), [{"__req__": req}])
                state = _LIVE_REQUESTS[req]
                req.output_ids.append(1)
                req.finished_reason = object()
                retained.append((req, state[-1], state[-2]))
                self.assertEqual(len(_LIVE_REQUESTS), 1)
            _reap_finished()
            self.assertEqual(len(_LIVE_REQUESTS), 0)
            self.assertTrue(all(s.journal._file.closed and not f.alive for _, s, f in retained))
            req = retained[0][0]
            req.finished_reason = None
            req.output_ids = []
            with self.assertRaisesRegex(ValueError, 'finished requests'):
                processor(torch.tensor([[-float('inf'), 0.]]), [{"__req__": req}])

    def test_unfinished_requests_keep_the_live_bound(self):
        from unittest.mock import Mock
        retained = [Request() for _ in range(32)]
        finalizers = []
        for req in retained:
            finalizer = Mock()
            finalizers.append(finalizer)
            _LIVE_REQUESTS[req] = (None, None, None, finalizer, None)
        _reap_finished()
        self.assertEqual(len(_LIVE_REQUESTS), 32)
        self.assertTrue(all(not f.called for f in finalizers))
        processor = object.__new__(KeyprintLogitsProcessor)
        with self.assertRaisesRegex(ValueError, 'request limit reached'):
            processor(torch.zeros((1, 2)), [{"__req__": Request()}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
