import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from testsite.core.integrity import file_sha256
from testsite.core.loader import DataLoader
from testsite.scripts.generate_silent_control import generate, BLOCK_FRAMES


class SilentControlTests(unittest.TestCase):
    def fixture(self, folder):
        records = []
        for i, (rate, channels, frames, subtype) in enumerate([
            (16000, 1, 317, 'PCM_16'), (44100, 2, BLOCK_FRAMES + 17, 'PCM_24'),
            (8000, 1, 1, 'PCM_U8'), (48000, 2, 513, 'FLOAT')
        ]):
            filename = f'{i}.wav'
            sf.write(str(folder / filename), np.full((frames, channels), .25), rate, subtype=subtype)
            records.append(dict(id=str(i), audio=filename, qa_prompt_version='fixed',
                _gt={'L1': 'active', 'L2': 'pulse', 'L3': 'CW'}, _meta={'test': i},
                conversations=[{'from': role, 'value': text} for role, text in [
                    ('human', 'Original question 1'), ('gpt', 'A'),
                    ('human', 'Original question 2'), ('gpt', 'A'),
                    ('human', 'Original question 3'), ('gpt', 'Original reference')]]))
        manifest = folder / 'input.jsonl'
        manifest.write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
        return manifest, records

    def test_exact_silence_and_annotations_and_original_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest, records = self.fixture(folder)
            hashes = {r['audio']: file_sha256(folder / r['audio']) for r in records}
            with contextlib.redirect_stdout(io.StringIO()):
                output = generate(manifest, folder, folder / 'control')
            silent = [json.loads(s) for s in (output / 'silent.jsonl').read_text().splitlines()]
            original = [json.loads(s) for s in (output / 'original.jsonl').read_text().splitlines()]
            for src, dst, orig in zip(records, silent, original):
                self.assertEqual({k: v for k, v in dst.items() if k != 'audio'},
                                 {k: v for k, v in src.items() if k != 'audio'})
                expected = sf.info(str(folder / src['audio']))
                samples, rate = sf.read(str(output / dst['audio']), always_2d=True)
                self.assertEqual(samples.shape, (expected.frames, expected.channels))
                self.assertEqual(rate, expected.samplerate)
                self.assertTrue(np.all(samples == 0))
                self.assertEqual(file_sha256(folder / src['audio']), hashes[src['audio']])
                self.assertEqual(Path(orig['audio']), folder / src['audio'])
            self.assertEqual(len(DataLoader({}).load(str(output / 'silent.jsonl'), str(output))), 4)
            self.assertEqual(len(DataLoader({}).load(str(output / 'original.jsonl'), str(output))), 4)
            self.assertEqual(json.loads((output / 'generation.json').read_text())['status'], 'complete')

    def test_missing_source_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest, records = self.fixture(folder)
            (folder / records[0]['audio']).unlink()
            with self.assertRaises(FileNotFoundError):
                generate(manifest, folder, folder / 'control')
            self.assertEqual(list(folder.glob('control_*')), [])


if __name__ == '__main__':
    unittest.main()
