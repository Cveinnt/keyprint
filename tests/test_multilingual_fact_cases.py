"""Model-free structural checks, not language or inference qualification."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MultilingualFactCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(
            (ROOT / "tools/multilingual_fact_cases.json").read_text(encoding="utf-8")
        )

    def test_case_identities_are_unique_and_safe(self):
        self.assertEqual(
            [(case["id"], case["language"]) for case in self.cases],
            [("workshop-th", "Thai"), ("workshop-id", "Indonesian"),
             ("workshop-de", "German")],
        )
        self.assertEqual(len({case["id"] for case in self.cases}), len(self.cases))
        for case in self.cases:
            self.assertRegex(case["id"], r"^[a-z0-9][a-z0-9_-]{0,63}$")

    def test_generate_only_harness_contract(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(case.get("action", "generate"), "generate")
                self.assertIsInstance(case["prompt"], str)
                self.assertTrue(case["prompt"].strip())
                self.assertLessEqual(len(case["prompt"]), 16000)
                self.assertIsInstance(case["review"], str)
                self.assertTrue(case["review"].strip())
                self.assertIs(type(case["max_tokens"]), int)
                self.assertGreaterEqual(case["max_tokens"], 1)
                self.assertLessEqual(case["max_tokens"], 1024)
                self.assertNotIn("max_words", case)
                self.assertNotIn("expected_json", case)
                self.assertNotIn("preserve", case)

    def test_same_explicit_source_obligations(self):
        facts = self.cases[0]["facts"]
        self.assertEqual(len(facts), 6)
        self.assertEqual(len(set(facts)), 6)
        self.assertTrue(all(isinstance(fact, str) and fact.strip() for fact in facts))
        for case in self.cases:
            self.assertEqual(case["facts"], facts)
            self.assertIn("six facts", case["review"])
            self.assertIn("ambiguous", case["review"])
            self.assertIn(case["language"], case["review"])

    def test_required_literals_are_requested_not_invented(self):
        expected = ["Luma", "Mira", "Narin", "18", "11", "7", "10:30"]
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["required_literals"], expected)
                for literal in expected:
                    self.assertIn(literal, case["prompt"])
                self.assertNotRegex(case["prompt"], r"https?://|[\w.+-]+@[\w.-]+")

    def test_does_not_replace_previous_quality_cases(self):
        existing = json.loads(
            (ROOT / "tools/quality_cases.json").read_text(encoding="utf-8")
        )
        self.assertFalse({case["id"] for case in existing}
                         & {case["id"] for case in self.cases})


if __name__ == "__main__":
    unittest.main()
