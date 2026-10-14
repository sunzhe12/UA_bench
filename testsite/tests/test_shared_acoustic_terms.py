"""Shared observations must not become cross-class contradictions."""
import unittest
from testsite.config import load_config
from testsite.core.scorer import Scorer
from testsite.core.shared_terminology import L3_PRECISE, L3_SHOULD, get_should_not


class SharedAcousticTermsTests(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(load_config())

    def test_generic_cues_are_compatible(self):
        cases = [(leaf, 'A chirp is audible.') for leaf in ('LFM', 'HFM')]
        cases += [(leaf, 'Symbol-wise phase-state transitions with phase reversal.')
                  for leaf in ('BPSK', 'QPSK')]
        cases += [(leaf, 'Approximately 180-degree phase transitions.')
                  for leaf in ('BPSK', 'QPSK')]
        cases += [(leaf, 'A narrowband structure is present.') for leaf in L3_SHOULD]
        for leaf, text in cases:
            with self.subTest(leaf=leaf, text=text):
                result = self.scorer.compute_reasoning_quality([text], [leaf])
                self.assertEqual(result.alignment_rate, 1)
                self.assertEqual(result.contradiction_rate, 0)
                self.assertEqual(result.term_stacking_rate, 0)

    def test_specific_conflicts_are_preserved(self):
        for leaf, text in [('HFM', 'A linear chirp.'),
                           ('LFM', 'A hyperbolic chirp.'),
                           ('BPSK', 'Four phase states.'),
                           ('QPSK', 'Two phase states.'),
                           ('2FSK', 'Four frequency states.'),
                           ('4FSK', 'Two frequency states.')]:
            with self.subTest(leaf=leaf):
                result = self.scorer.compute_reasoning_quality([text], [leaf])
                self.assertEqual(result.concept_confusion_rate, 1)
                self.assertEqual(result.alignment_rate, 0)

    def test_shared_cue_does_not_hide_specific_conflict(self):
        result = self.scorer.compute_reasoning_quality(
            ['Two phase states and narrowband structure.'], ['BPSK'])
        self.assertEqual(result.alignment_rate, 1)
        result = self.scorer.compute_reasoning_quality(
            ['Four phase states and narrowband structure.'], ['BPSK'])
        self.assertEqual(result.term_stacking_rate, 1)

    def test_inventory_consistency(self):
        for leaf, terms in L3_SHOULD.items():
            with self.subTest(leaf=leaf):
                self.assertTrue(set(L3_PRECISE[leaf]) <= terms)
                self.assertFalse(terms & get_should_not(leaf))
                self.assertNotIn('narrowband', get_should_not(leaf))


if __name__ == '__main__':
    unittest.main()
