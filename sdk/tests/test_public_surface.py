"""Installed-wheel regressions; supplied fixtures only, no model downloads."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import keyprint_v3
from keyprint_v3 import DurableJournal, PublicCandidate


KEY = bytes(range(32))
CORE = "0f78c82a544c496644d4e1c9b13809781a97baf4451b190fa8e2bcfac4f36477"
FACADE = "7c1266a980dddaae26c91078b010c4aab6c24fca290dc96cbe23adb5b9bf4097"


def head(*tokens):
    result = np.full((1, 151936), -np.inf, dtype=np.float32)
    result[0, list(tokens)] = 0
    return result


class FakeBackend:
    array = staticmethod(np.array)
    int32 = np.int32
    float32 = np.float32
    eval = staticmethod(lambda value: None)


class PublicSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = PublicCandidate()

    def assert_report(self, report, kind):
        self.assertEqual(report["kind"], kind)
        self.assertIsNone(report["verdict"])
        self.assertFalse(report["calibration"]["deployment_calibrated"])
        self.assertTrue(report["interpretation"])
        json.dumps(report, allow_nan=False)

    def test_public_version_and_scientific_identities(self):
        self.assertEqual(keyprint_v3.__version__, "0.0.4rc4")
        self.assertIn("DurableJournal", keyprint_v3.__all__)
        self.assertEqual(self.candidate.core_identity["runtime_profile_sha256"], CORE)
        self.assertEqual(self.candidate.identity["facade_profile_sha256"], FACADE)

    def test_supported_journal_with_supplied_model(self):
        calls = []

        def model(ids, *, cache):
            return head(32).reshape(1, 1, 151936).repeat(ids.shape[1], axis=1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.jsonl"
            with DurableJournal(path) as journal:
                report = self.candidate.run_response(
                    model, [32], key=KEY, condition="ordinary",
                    random_bits=lambda bits: 0, journal=journal,
                    reserve=lambda action, metadata: calls.append(action),
                    max_tokens=2, backend=FakeBackend,
                    cache_factory=lambda model: [],
                )
            self.assert_report(report, "generation_trace")
            self.assertEqual(report["rendered_carriers"]["visible_text"], "AA")
            self.assertEqual(report["payload"]["committed_token_ids"], [32, 32])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-1]["event"]["outcome"], "length")
            self.assertIn("model_forward", calls)
            with self.assertRaises(FileExistsError):
                DurableJournal(path)

    def test_shared_filter_rejects_dtype_without_randomness(self):
        requests = []
        with self.candidate.pipeline(KEY, condition="ordinary") as pipeline:
            report = pipeline.step(head(32, 33).astype(np.float64), lambda bits: requests.append(bits))
            self.assert_report(report, "error")
            self.assertEqual(report["payload"]["committed_tokens"], 0)
            self.assertEqual(requests, [])
            self.assertEqual(pipeline.step(head(32), lambda bits: 0), report)

    def test_literal_availability_and_invalid_key(self):
        for text in ("The sky is blue.", "中文 😀 café", "", " \n", "e\u0301"):
            report = self.candidate.score_literal(text, KEY)
            self.assert_report(report, "literal_diagnostic")
        self.assert_report(self.candidate.score_literal("hello", b"invalid"), "error")

    def test_generated_decomposed_accent_retained(self):
        # Qwen's fixed single-byte vocabulary represents e + UTF-8 CC 81.
        # These explicit token IDs are fixture data, not private API access.
        with self.candidate.pipeline(KEY, condition="ordinary") as pipeline:
            for token in (68, 136, 223):
                self.assert_report(pipeline.step(head(token), lambda bits: 0), "generation_trace")
            report = pipeline.finish()
        self.assert_report(report, "generation_trace")
        self.assertEqual(report["rendered_carriers"]["visible_text"], "e\u0301")
        self.assertEqual(report["literal_replay_status"][0]["availability"], "unavailable")
        self.assertIn("normalizes", report["literal_replay_status"][0]["reason"])
        self.assertIsNone(report["payload"]["literal_diagnostics"][0]["events"])

    def test_channel_routes_both_conditions(self):
        for condition in ("ordinary", "marked"):
            with self.subTest(condition=condition):
                with self.candidate.pipeline(KEY, condition=condition, allow_thinking=True, allow_tools=True) as pipeline:
                    for token in (151667, 32, 151668, 151657, 33, 151658, 34, 151645):
                        self.assert_report(pipeline.step(head(token), lambda bits: 0), "generation_trace")
                    report = pipeline.finish()
                self.assertEqual(report["payload"]["completion"], "eos")
                self.assertEqual(report["rendered_carriers"]["visible_text"], "C")
                self.assertEqual(report["rendered_carriers"]["reasoning_text"], "A")
                self.assertEqual(report["rendered_carriers"]["tool_texts"], ["B"])

    def test_cli_commands(self):
        for args in (["demo"], ["demo", "--condition", "ordinary"], ["verify"], ["score-fixture"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-m", "keyprint_v3", *args],
                    input="The sky is blue.\n", text=True, capture_output=True, cwd=directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertIn(report["kind"], ("generation_trace", "verification", "literal_diagnostic"))
                self.assertIsNone(report["verdict"])
                self.assertTrue(report["interpretation"])

    def test_corrupt_bundle_fails_closed_before_json(self):
        # Mutate only a disposable copy, never the installed or frozen package.
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "keyprint_v3"
            shutil.copytree(Path(keyprint_v3.__file__).parent, copy)
            source = copy / "_bundle/research/keyprint_reporting_v2.py"
            source.write_bytes(source.read_bytes() + b"\n# disposable corruption regression\n")
            result = subprocess.run([sys.executable, "-m", "keyprint_v3", "verify"],
                text=True, capture_output=True, cwd=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("Keyprint bundled source integrity mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
