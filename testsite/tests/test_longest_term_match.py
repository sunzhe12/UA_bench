"""Regression tests for overlap resolution without changing metric formulas."""
import unittest

from testsite.config import load_config
from testsite.core.scorer import Scorer


class LongestTermMatchTests(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(load_config())

    def test_nested_negative_term_is_suppressed_across_both_sets(self):
        result = self.scorer.compute_reasoning_quality(
            ['I hear a hyperbolic chirp with a curved time-frequency trajectory.'],
            ['HFM'])
        self.assertEqual(result.alignment_rate, 1)
        self.assertEqual(result.contradiction_rate, 0)
        self.assertEqual(result.term_stacking_rate, 0)

    def test_independent_conflicting_phrase_still_counts(self):
        result = self.scorer.compute_reasoning_quality(
            ['I hear a hyperbolic chirp and a linear chirp.'], ['HFM'])
        self.assertEqual(result.alignment_rate, 0)
        self.assertEqual(result.contradiction_rate, 1)
        self.assertEqual(result.term_stacking_rate, 1)

    def test_short_term_elsewhere_is_not_globally_suppressed(self):
        hits = self.scorer._longest_term_matches(
            'a hyperbolic chirp followed by another chirp',
            {'hyperbolic chirp', 'chirp'})
        self.assertEqual(hits, {'hyperbolic chirp', 'chirp'})

    def test_case_boundaries_and_adjacent_occurrences(self):
        hits = self.scorer._longest_term_matches(
            'HYPERBOLIC CHIRP,linear chirp; chirping',
            {'hyperbolic chirp', 'linear chirp', 'chirp'})
        self.assertEqual(hits, {'hyperbolic chirp', 'linear chirp'})
        self.assertEqual(self.scorer._longest_term_matches('chirping', {'chirp'}), set())

    def test_partial_overlap_and_equal_length_tie_are_deterministic(self):
        self.assertEqual(self.scorer._longest_term_matches(
            'alpha beta gamma', {'alpha beta', 'beta gamma'}), {'alpha beta'})
        self.assertEqual(self.scorer._longest_term_matches(
            'alpha beta gamma delta', {'alpha beta', 'beta gamma delta'}),
            {'beta gamma delta'})

    def test_original_category_formulas_and_denominator_are_preserved(self):
        result = self.scorer.compute_reasoning_quality(
            ['hyperbolic chirp', 'linear chirp',
             'hyperbolic chirp and linear chirp', 'insufficient evidence'],
            ['HFM'] * 4)
        self.assertEqual(result.num_samples, 4)
        self.assertEqual(result.alignment_rate, .25)
        self.assertEqual(result.concept_confusion_rate, .25)
        self.assertEqual(result.term_stacking_rate, .25)
        self.assertEqual(result.vague_rate, .25)
        self.assertEqual(result.contradiction_rate, .5)


if __name__ == '__main__':
    unittest.main()
