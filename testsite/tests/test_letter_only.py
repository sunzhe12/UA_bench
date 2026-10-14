"""Strict classification format, fixed paths, and gate regression tests."""
import unittest

from testsite.core.source_label_prompts import OPTION_ONLY_INSTRUCTION
from testsite.config import load_config
from testsite.core.loader import EvalSample
from testsite.core.parser import OutputParser
from testsite.eval.multi_turn import MultiTurnEvaluator


class LetterOnlyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.parser = OutputParser(self.config)
        self.prompts = self.config['prompts']['three_turn']

    def test_all_options_and_templates(self):
        for branch, leaves, l1 in [
            ('turn2_active', ['CW', 'LFM', 'HFM', '2FSK', '4FSK', 'BPSK', 'QPSK', 'OFDM'], 'active'),
            ('turn2_passive', ['cargo', 'cruise', 'fishing', 'warship', 'underwater_target'], 'passive'),
        ]:
            for prompt in self.prompts[branch]:
                self.assertTrue(prompt.endswith(OPTION_ONLY_INSTRUCTION))
                for i, leaf in enumerate(leaves):
                    answer = ' \n' + chr(65 + i) + '\t'
                    pred = self.parser.parse('s', answer, prompt)
                    expected_l2 = 'ship_noise' if l1 == 'passive' else ('pulse' if i < 3 else 'communication')
                    self.assertEqual((pred.L1, pred.L2, pred.L3), (l1, expected_l2, leaf))
                    self.assertEqual(pred.raw_output, answer)
                    self.assertEqual(pred.parse_status, 'valid_option')

    def test_ascii_lowercase_equivalence_and_raw_output(self):
        for key in ['turn1', 'turn2_active', 'turn2_passive']:
            for prompt in self.prompts[key]:
                import re
                letters = re.findall(r'^([A-Z])\. ', prompt, re.MULTILINE)
                self.assertTrue(letters)
                for letter in letters:
                    raw = ' \n' + letter.lower() + '\t'
                    upper = self.parser.parse('s', letter, prompt)
                    lower = self.parser.parse('s', raw, prompt)
                    self.assertEqual((lower.L1, lower.L2, lower.L3),
                                     (upper.L1, upper.L2, upper.L3))
                    self.assertEqual(lower.parse_status, 'valid_option')
                    self.assertEqual(lower.raw_output, raw)
                    if key == 'turn1':
                        self.assertEqual(self.parser.parse_turn1('s', raw, prompt).L1, upper.L1)

    def test_rejects_prose_labels_multiple_choices_and_bad_format(self):
        for answer in ['BPSK', 'CW', '2FSK', 'cargo', 'cruise', 'A or B',
                       'Answer: B', 'Option B', 'A.', '(A)', '', 'Z', 'z', 'Ａ', 'ａ', 'ß', 'a b', 'a.',
                       'I cannot decide between active and passive.']:
            for key in ['turn1', 'turn2_active', 'turn2_passive']:
                pred = self.parser.parse('s', answer, self.prompts[key][0])
                self.assertEqual((pred.L1, pred.L2, pred.L3), ('unknown',) * 3)
                self.assertEqual(pred.parse_status, 'invalid_format')
        self.assertEqual(self.parser.parse('s', 'F', self.prompts['turn2_passive'][0]).L3, 'unknown')

    def test_invalid_answers_gate_and_score_without_keyword_credit(self):
        class Answers:
            def __init__(self, first, second):
                self.first, self.second, self.chat_calls = first, second, 0

            def generate_batch(self, paths, prompts):
                return [self.first] * len(paths)

            def chat_batch(self, paths, histories):
                self.chat_calls += 1
                return [self.second if self.chat_calls == 1 else 'Explanation.'] * len(paths)

        sample = EvalSample('s', 'unused.wav', {'L1': 'active', 'L2': 'communication', 'L3': 'BPSK'}, questions=(self.prompts['turn1'][0], self.prompts['turn2_active'][0], self.prompts['turn3']))
        for first, second, expected, calls in [('active', 'F', 0, 0), ('A', 'BPSK', 0, 2), ('A', 'F', 1, 2), ('a', 'f', 1, 2)]:
            model = Answers(first, second)
            metrics, _, results = MultiTurnEvaluator(self.config, model).evaluate([sample])
            self.assertEqual(metrics.l3.accuracy, expected)
            from testsite.reporting.report import ReportGenerator
            exported = ReportGenerator._to_dict(metrics)
            expected_keys = {
                'total_samples', 'l1_accuracy', 'l2_accuracy', 'l3_accuracy',
                'l3_macro_f1', 'l2_given_l1', 'l3_given_l2', 'parse_tier_dist',
                'l3_per_class', 'l3_confusion_matrix', 'l3_confusion_labels',
            }
            self.assertEqual(set(exported), expected_keys)
            self.assertEqual(model.chat_calls, calls)
            self.assertEqual(results[0].cascade_skipped, calls == 0)


if __name__ == '__main__':
    unittest.main()
