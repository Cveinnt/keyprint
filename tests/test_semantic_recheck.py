"""The recount must reject missing attempts and altered review/output joins."""
import copy
import unittest
from tools.recheck_semantic_evidence import recount


class SemanticRecheckTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(case='case', key_slot=0, condition=c, review_id=c,
                          text='Approved only after review.', completion='eos')
                     for c in ('ordinary', 'marked')]
        self.plan = {'schedule': copy.deepcopy(self.rows)}
        self.blind = [dict(review_id=r['review_id'], text=r['text'], completion='eos',
                           case={'id': 'case', 'facts': ['approval requires review']})
                      for r in self.rows]
        self.ratings = [dict(review_id=r['review_id'], fact_checks=[True],
                             no_unsupported_claims=True, language_pass=True,
                             reason='The condition is retained.') for r in self.rows]

    def check(self):
        return recount(self.plan, self.rows, self.ratings, self.blind)

    def test_complete_pair_and_failed_meaning_are_retained(self):
        self.ratings[1]['fact_checks'] = [False]
        result = self.check()
        self.assertEqual(result['paired_content'], {'ordinary_only': 1})
        self.assertEqual(result['groups']['marked']['language_pass'], 1)
        self.assertEqual(result['groups']['marked']['content_pass'], 0)

    def test_missing_attempt_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'attempt'):
            self.check()

    def test_duplicate_rating_rejected(self):
        self.ratings[1] = copy.deepcopy(self.ratings[0])
        with self.assertRaisesRegex(ValueError, 'review'):
            self.check()

    def test_changed_text_rejected(self):
        self.rows[1]['text'] = 'Approved now.'
        with self.assertRaisesRegex(ValueError, 'output differs'):
            self.check()

    def test_missing_fact_rejected(self):
        self.ratings[1]['fact_checks'] = []
        with self.assertRaisesRegex(ValueError, 'fact review'):
            self.check()

    def test_truthy_nonboolean_rejected(self):
        self.ratings[1]['language_pass'] = 'false'
        with self.assertRaisesRegex(ValueError, 'Invalid semantic rating'):
            self.check()
