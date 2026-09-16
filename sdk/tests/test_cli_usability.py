"""Installed entrypoint checks; frozen numerical contracts remain in the API suite."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class CommandTests(unittest.TestCase):
    def invoke(self, *args):
        command = Path(sys.executable).with_name("keyprint")
        return subprocess.run([str(command), *args], capture_output=True, text=True, check=False)

    def test_no_arguments_show_help(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: keyprint", result.stdout)
        self.assertIn("doctor", result.stdout)

    def test_demo_is_readable_and_opt_in_json_retains_reports(self):
        human = self.invoke("demo")
        self.assertEqual(human.returncode, 0, human.stderr)
        self.assertIn("BDABCAAD", human.stdout)
        self.assertIn("CDAABDCB", human.stdout)
        self.assertLess(len(human.stdout), 600)
        machine = self.invoke("demo", "--json")
        self.assertEqual(machine.returncode, 0, machine.stderr)
        artifact = json.loads(machine.stdout)
        for report in artifact["reports"].values():
            self.assertEqual(report["kind"], "generation_trace")
            self.assertIsNone(report["verdict"])

    def test_verify_and_doctor(self):
        for command in ("verify", "doctor"):
            result = self.invoke(command, "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "pass")

    def test_unknown_option_fails(self):
        result = self.invoke("demo", "--secret-key", "should-not-be-accepted")
        self.assertNotEqual(result.returncode, 0)

    def test_generate_help_and_missing_model(self):
        help_result = self.invoke("generate", "--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--prompt", help_result.stdout)
        result = self.invoke("generate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--model", result.stderr)
