"""CPU end-to-end tests for version-locked aggregation and silence comparison."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from testsite.config import load_config
from testsite.core.dataset_record import LEAF_PARENTS
from testsite.core.protocol import identity_signature
from testsite.scripts.generate_silent_control import generate
from testsite.scripts.merge_predictions import aggregate
from testsite.scripts.compare_silent_control import compare, read_rows


class ReleaseProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        prompts = load_config()['prompts']['three_turn']
        records = []
        for i, (leaf, (l1, l2)) in enumerate(LEAF_PARENTS.items()):
            audio = f'{i}.wav'
            sf.write(cls.base / audio, np.full(80, .25), 8000, subtype='PCM_16')
            qs = [prompts['turn1'][0], prompts['turn2_'+l1][0], prompts['turn3']]
            records.append({'id': str(i), 'audio': audio,
                            '_gt': {'L1': l1, 'L2': l2, 'L3': leaf},
                            'conversations': [{'from': 'human', 'value': q} for q in qs]})
        manifest = cls.base / 'input.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.control = generate(manifest, cls.base, cls.base/'control')
        cls.paths = {}
        for condition in ('original', 'silent'):
            output = cls.base / condition
            result = subprocess.run([sys.executable, '-B', '-m', 'testsite.scripts.run_eval',
                '--data', str(cls.control/(condition+'.jsonl')), '--audio-root', str(cls.control),
                '--backend', 'mock', '--output-dir', str(output)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.paths[condition] = (output/'predictions_shard_000.jsonl', output/'protocol_shard_000.json')

    def test_aggregate_matches_original_report(self):
        predictions, protocol = self.paths['original']
        metrics, _ = aggregate(read_rows(predictions), self.control/'original.jsonl', [protocol])
        report = json.loads(next(predictions.parent.glob('eval_metrics_*.json')).read_text())
        for key, value in report.items():
            self.assertEqual(metrics[key], value)

    def test_complete_multishard_protocol_is_required(self):
        rows, protocols = [], []
        for index in range(2):
            output = self.base / f'shard_{index}'
            result = subprocess.run([sys.executable, '-B', '-m', 'testsite.scripts.run_eval',
                '--data', str(self.control/'original.jsonl'), '--audio-root', str(self.control),
                '--backend', 'mock', '--num-shards', '2', '--shard-index', str(index),
                '--run-id', 'paired-shard-test', '--output-dir', str(output)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows.extend(read_rows(output/f'predictions_shard_{index:03d}.jsonl'))
            protocols.append(output/f'protocol_shard_{index:03d}.json')
        metrics, _ = aggregate(rows, self.control/'original.jsonl', protocols)
        self.assertEqual(metrics['total_samples'], 13)
        with self.assertRaisesRegex(ValueError, 'incomplete protocol'):
            aggregate(rows, self.control/'original.jsonl', protocols[:1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            aggregate(rows, self.control/'original.jsonl', [protocols[0], protocols[0]])

    def test_bad_protocols_fail(self):
        predictions, protocol = self.paths['original']
        rows = read_rows(predictions)
        saved = json.loads(protocol.read_text())
        with self.assertRaises(ValueError):
            aggregate(rows, self.control/'original.jsonl', [])
        for case in ('version', 'signature', 'missing_shard', 'taxonomy'):
            value = deepcopy(saved)
            if case == 'version':
                value['run_identity']['implementation']['testsite/core/shared_terminology.py'] = 'changed'
            elif case == 'signature':
                value['run_signature'] = 'changed'
            elif case == 'missing_shard':
                value['num_shards'] = 2
            else:
                value['taxonomy'] = {}
            target = self.base/'bad_protocol.json'
            target.write_text(json.dumps(value))
            with self.subTest(case=case), self.assertRaises(ValueError):
                aggregate(rows, self.control/'original.jsonl', [target])

    def test_uses_saved_taxonomy_not_default(self):
        predictions, protocol = self.paths['original']
        rows = read_rows(predictions)
        value = json.loads(protocol.read_text())
        tax = value['run_identity']['taxonomy']
        tax['L3']['classes'][0]['name'] = 'Archived display name'
        value['taxonomy'] = deepcopy(tax)
        value['run_signature'] = identity_signature(value['run_identity'])
        for row in rows:
            row['run_signature'] = value['run_signature']
        target = self.base/'saved_taxonomy.json'
        target.write_text(json.dumps(value))
        metrics, _ = aggregate(rows, self.control/'original.jsonl', [target])
        self.assertIn('Archived display name', metrics['l3_per_class'])

    def test_silent_comparison_complete_and_rejects_drift(self):
        a, pa = self.paths['original']; b, pb = self.paths['silent']
        output = compare(self.control, a, b, [pa], [pb], self.base/'comparison')
        report = json.loads((output/'comparison.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(len(read_rows(output/'paired_predictions.jsonl')), 13)
        for key, value in report['delta'].items():
            self.assertEqual(value, report['original'][key]-report['silent'][key])
        value = json.loads(pb.read_text())
        value['run_identity']['model']['temperature'] = 987
        value['run_signature'] = identity_signature(value['run_identity'])
        rows = read_rows(b)
        for row in rows:
            row['run_signature'] = value['run_signature']
        badp, badr = self.base/'changed_protocol.json', self.base/'changed_predictions.jsonl'
        badp.write_text(json.dumps(value)); badr.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'different model'):
            compare(self.control, a, badr, [pa], [badp], self.base/'bad_comparison')
        wav = self.control/'audio/000000.wav'
        original_bytes = wav.read_bytes()
        try:
            wav.write_bytes(original_bytes+b'changed')
            with self.assertRaisesRegex(ValueError, 'audio changed'):
                compare(self.control, a, b, [pa], [pb], self.base/'bad_audio')
        finally:
            wav.write_bytes(original_bytes)
