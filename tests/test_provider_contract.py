"""Model-free checks of supported versus unexpectedly broken rejection behavior."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "provider_contract", Path(__file__).resolve().parents[1] / "tools/provider_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class Blocked(Exception):
    pass


class ProviderContractTests(unittest.TestCase):
    def test_expected_rejection_is_not_a_compatibility_failure(self):
        calls = []

        def reject():
            calls.append(1)
            raise Blocked()

        result, failed = contract.check_rewrite_rejection(reject, Blocked)
        self.assertFalse(failed)
        self.assertEqual(result["status"], "expected_rejection")
        self.assertEqual(calls, [1])

    def test_unrelated_errors_still_fail(self):
        def broken():
            raise RuntimeError("failure")

        result, failed = contract.check_rewrite_rejection(broken, Blocked)
        self.assertTrue(failed)
        self.assertEqual(result["status"], "unexpected_error")
        self.assertEqual(result["error_type"], "RuntimeError")

    def test_silent_rewrite_success_is_a_failure(self):
        result, failed = contract.check_rewrite_rejection(lambda: "changed text", Blocked)
        self.assertTrue(failed)
        self.assertEqual(result["status"], "unexpected_success")


if __name__ == "__main__":
    unittest.main()
