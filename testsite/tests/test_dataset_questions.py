"""Published questions are immutable inputs; reference answers never reach models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from testsite.config import load_config
from testsite.core.loader import DataLoader, EvalSample
from testsite.eval.multi_turn import MultiTurnEvaluator


class Recorder:
    def __init__(self, wrong=()):
        self.wrong = set(wrong)
        self.seen = []

    def generate_batch(self, paths, prompts):
        self.seen.extend(zip(paths, prompts))
        return ['B' if Path(p).stem in self.wrong else 'A' for p in paths]

    def chat_batch(self, paths, histories):
        self.seen.extend(zip(paths, deepcopy(histories)))
        return ['F' if len(h) == 3 else 'two phase states' for h in histories]

    def generate(self, path, prompt, **kwargs):
        return self.generate_batch([path], [prompt])[0]

    def chat(self, path, history, **kwargs):
        return self.chat_batch([path], [history])[0]


class DatasetQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.templates = self.config.pop('prompts')['three_turn']
        self.loader = DataLoader(self.config)

    def record(self, sid):
        qs = (f'Record {sid}\n' + self.templates['turn1'][0],
              self.templates['turn2_active'][0].replace('{L1}', 'actively transmitted'),
              f'Explain record {sid}.')
        turns = []
        for q in qs:
            turns += [{'from': 'human', 'value': q},
                      {'from': 'gpt', 'value': 'SECRET_REFERENCE_NEVER_SEND'}]
        return {'id': sid, 'audio': sid + '.wav',
                '_gt': {'L1': 'active', 'L2': 'communication', 'L3': 'BPSK'},
                'qa_prompt_version': 'dataset-version', 'conversations': turns}

    def samples(self):
        return [self.loader._build_sample(self.record(str(i)), Path('.')) for i in range(8)]

    def test_loader_keeps_exact_questions_and_not_reference_answers(self):
        record = self.record('0')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            path.write_text(json.dumps(record) + '\n', encoding='utf-8')
            (Path(tmp) / record['audio']).write_bytes(b'test fixture')
            sample = self.loader.load(str(path), tmp)[0]
        self.assertEqual(sample.questions, tuple(t['value'] for t in record['conversations'][::2]))
        self.assertEqual(sample.qa_prompt_version, 'dataset-version')
        self.assertNotIn('SECRET_REFERENCE', repr(sample))

    def test_questions_independent_of_batch_shards_and_routing(self):
        samples = self.samples()
        for batch, wrong in [(1, ()), (4, ()), (4, ('0', '2', '3'))]:
            cfg = deepcopy(self.config)
            cfg['model']['batch_size'] = batch
            for subset in [samples, samples[::2], samples[1::2]]:
                model = Recorder(wrong)
                _, _, results = MultiTurnEvaluator(cfg, model).evaluate(subset)
                for r in results:
                    self.assertEqual(r.turn1_prompt, r.sample.questions[0])
                    if not r.cascade_skipped:
                        self.assertEqual(r.turn2_prompt, r.sample.questions[1])
                        self.assertEqual(r.turn3_prompt, r.sample.questions[2])
                self.assertNotIn('SECRET_REFERENCE', repr(model.seen))
                for _, history in model.seen:
                    if isinstance(history, list):
                        self.assertEqual(history[1]['value'], 'A')
                        if len(history) == 5:
                            self.assertEqual(history[3]['value'], 'F')

    def test_all_three_turns_use_dataset_questions_verbatim(self):
        model = Recorder()
        _, _, results = MultiTurnEvaluator(self.config, model).evaluate(self.samples())
        checked = 0
        for r in results:
            if r.cascade_skipped:
                continue
            self.assertEqual((r.turn1_prompt, r.turn2_prompt, r.turn3_prompt), r.sample.questions)
            checked += 1
        self.assertGreater(checked, 0)
        self.assertNotIn('SECRET_REFERENCE', repr(model.seen))

    def test_missing_questions_fail_instead_of_using_config(self):
        for turns in [None, [], [{'from': 'human', 'value': 'One question'}]]:
            record = self.record('broken')
            record['conversations'] = turns
            with self.assertRaisesRegex(ValueError, 'broken'):
                self.loader._build_sample(record, Path('.'))
        model = Recorder()
        with self.assertRaises(ValueError):
            MultiTurnEvaluator(self.config, model).evaluate([
                EvalSample('broken', 'unused', {'L1': 'active'})])
        self.assertEqual(model.seen, [])


if __name__ == '__main__':
    unittest.main()
