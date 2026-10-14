from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from testsite.core.integrity import file_sha256, read_manifest, validate_predictions
from testsite.eval.multi_turn import MultiTurnEvaluator
from testsite.tests import test_dataset_questions as fixtures


class IntegrityGuardsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DatasetQuestionsTests()
        self.fixture.setUp()

    def manifest(self, folder):
        records = [self.fixture.record(str(i)) for i in range(2)]
        path = folder / 'data.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
        return path, records

    def predictions(self, path, records):
        return [dict(sample_id=r['id'], gt=r['_gt'], run_signature='same-run',
                     manifest_sha256=file_sha256(path), qa_prompt_version=r['qa_prompt_version'],
                     turn1_pred='active', cascade_skipped=False,
                     turn1_prompt=r['conversations'][0]['value'],
                     turn2_prompt=r['conversations'][2]['value'],
                     turn3_prompt=r['conversations'][4]['value']) for r in records]

    def test_manifest_rejects_bad_json_duplicate_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, records = self.manifest(Path(tmp))
            for text in ['{broken', json.dumps(records[0]) + '\n' + json.dumps(records[0]),
                         json.dumps({'id': 'x'}), '[]', '']:
                p.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_manifest(p)

    def test_missing_audio_and_unloadable_gt_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, records = self.manifest(folder)
            with self.assertRaises(FileNotFoundError):
                self.fixture.loader.load(str(path), tmp)
            for record in records:
                (folder / record['audio']).write_bytes(b'fixture')
            self.assertEqual(len(self.fixture.loader.load(str(path), tmp)), 2)
            with patch.object(self.fixture.loader, '_build_sample', return_value=None):
                with self.assertRaises(ValueError):
                    self.fixture.loader.load(str(path), tmp)

    def test_each_turn_rejects_incomplete_backend_output(self):
        for bad_turn in (1, 2, 3):
            class Short(fixtures.Recorder):
                calls = 0

                def generate_batch(self, paths, prompts):
                    return [] if bad_turn == 1 else super().generate_batch(paths, prompts)

                def chat_batch(self, paths, histories):
                    self.calls += 1
                    return [] if self.calls + 1 == bad_turn else super().chat_batch(paths, histories)

            with self.assertRaisesRegex(ValueError, f'Turn {bad_turn}'):
                MultiTurnEvaluator(self.fixture.config, Short()).evaluate(self.fixture.samples())

    def test_merge_rejects_duplicate_missing_extra_and_mixed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, records = self.manifest(Path(tmp))
            rows = self.predictions(path, records)
            self.assertEqual(validate_predictions(rows, path), file_sha256(path))
            cases = [rows + [rows[0]], rows[:1]]
            for field, value in [('sample_id', 'extra'), ('run_signature', 'different'),
                                 ('manifest_sha256', 'wrong'), ('turn2_prompt', 'changed'),
                                 ('gt', {'L1': 'passive'}), ('cascade_skipped', True)]:
                changed = deepcopy(rows)
                changed[1][field] = value
                cases.append(changed)
            for case in cases:
                with self.assertRaises(ValueError):
                    validate_predictions(case, path)


if __name__ == '__main__':
    unittest.main()
