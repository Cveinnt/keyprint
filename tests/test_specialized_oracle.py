import json
from pathlib import Path
import unittest
from tools.specialized_oracle import evaluate

CASES={c['id']:c for c in json.loads((Path(__file__).resolve().parents[1]/'tools/specialized_cases.json').read_text())}


class SpecializedOracleTests(unittest.TestCase):
    def test_reference_solutions(self):
        for case in CASES.values():
            with self.subTest(case=case['id']):
                self.assertEqual(evaluate(case['reference'],case)['status'],'pass')

    def test_lost_negation(self):
        c=CASES['python_release']
        self.assertEqual(evaluate(c['reference'].replace('not blocked','blocked'),c)['status'],'fail')

    def test_threshold_boundary(self):
        c=CASES['python_shipping']
        self.assertEqual(evaluate(c['reference'].replace('>= 5000','> 5000'),c)['status'],'fail')

    def test_code_cannot_run_imports_or_calls(self):
        c=CASES['python_release']
        for body in ['__import__("os").system("false")','open("/tmp/keyprint-unsafe", "w")',
                     'approved.__class__','[0] * 100000000000']:
            text='def can_release(approved, tests_passed, blocked):\n    return '+body
            self.assertEqual(evaluate(text,c)['status'],'unsupported_or_invalid')

    def test_sql_null_is_not_approval(self):
        c=CASES['sql_approval']
        text='SELECT id FROM orders WHERE coalesce(approved,1)=1 AND coalesce(blocked,0)=0 ORDER BY id'
        self.assertEqual(evaluate(text,c)['status'],'fail')

    def test_sql_cannot_write_or_attach(self):
        c=CASES['sql_approval']
        for text in ['DELETE FROM orders','ATTACH DATABASE "/tmp/keyprint-unsafe" AS other',
                     'SELECT load_extension("anything")','PRAGMA database_list']:
            self.assertEqual(evaluate(text,c)['status'],'unsupported_or_invalid')

    def test_sql_filter_before_aggregation_changes_meaning(self):
        c=CASES['sql_inventory']
        text=c['reference'].replace('GROUP BY','WHERE on_hand > reserved GROUP BY')
        self.assertEqual(evaluate(text,c)['status'],'fail')

    def test_json_exact_types_and_duplicate_keys(self):
        c=CASES['json_release']
        for old,new in [(':false',':0'),(':true',':"true"')]:
            self.assertEqual(evaluate(c['reference'].replace(old,new),c)['status'],'fail')
        duplicate=c['reference'][:-1]+',"approved":true}'
        self.assertEqual(evaluate(duplicate,c)['status'],'unsupported_or_invalid')

    def test_csv_timezone_and_role_are_required(self):
        for name,old,new in [('csv_timezone','Asia/Tokyo','UTC'),
                             ('csv_attribution','approver','recipient')]:
            c=CASES[name]
            self.assertEqual(evaluate(c['reference'].replace(old,new),c)['status'],'fail')

    def test_no_silent_markdown_cleanup(self):
        for c in CASES.values():
            self.assertNotEqual(evaluate('```\n'+c['reference']+'\n```',c)['status'],'pass')
