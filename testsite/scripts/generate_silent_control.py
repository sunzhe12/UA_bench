"""Build paired original/silent manifests with exact WAV frame/rate/channel parity.

Run from the repository root:
python -m testsite.scripts.generate_silent_control --data DATA.jsonl --audio-root AUDIO_ROOT
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from testsite.core.integrity import file_sha256, read_manifest
from testsite.core.loader import DataLoader


BLOCK_FRAMES = 65536
LOSSLESS_SUBTYPES = {'PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'}


def new_directory(prefix):
    prefix = Path(prefix)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for index in range(1000):
        candidate = prefix.with_name(prefix.name + '_' + stamp + (f'_{index}' if index else ''))
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError('Could not reserve a fresh silent-control directory')


def write_silence(path, info):
    """Write exact frame counts without normalization or resampling."""
    with sf.SoundFile(str(path), mode='w', samplerate=info.samplerate,
                      channels=info.channels, format='WAV', subtype=info.subtype) as output:
        zeros = np.zeros((min(BLOCK_FRAMES, info.frames), info.channels), dtype=np.float64)
        remaining = info.frames
        while remaining:
            count = min(remaining, len(zeros))
            output.write(zeros[:count])
            remaining -= count


def verify_silence(path, reference):
    actual = sf.info(str(path))
    for field in ('samplerate', 'frames', 'channels', 'subtype'):
        if getattr(actual, field) != getattr(reference, field):
            raise ValueError(f'{path}: mismatched {field}')
    count = 0
    with sf.SoundFile(str(path)) as stream:
        for block in stream.blocks(blocksize=BLOCK_FRAMES, dtype='float64', always_2d=True):
            count += len(block)
            if not np.all(block == 0):
                raise ValueError(f'{path}: nonzero samples in silent control')
    if count != reference.frames:
        raise ValueError(f'{path}: decoded frame count mismatch')


def generate(data, audio_root, output_prefix):
    data = Path(data).resolve()
    audio_root = Path(audio_root).resolve()
    records = read_manifest(data)
    loader = DataLoader({})
    prepared = []
    # Validate all source records before creating any outputs.
    for record in records:
        sample = loader._build_sample(record, audio_root)
        if sample is None:
            raise ValueError(f"{record['id']}: cannot recover ground truth")
        if any(not sample.gt.get(level) or sample.gt[level] == 'unknown' for level in ('L1', 'L2', 'L3')):
            raise ValueError(f"{record['id']}: incomplete ground truth")
        source = Path(sample.audio_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{record['id']}: missing source audio {source}")
        info = sf.info(str(source))
        if info.format not in ('WAV', 'WAVEX', 'RF64') or info.subtype not in LOSSLESS_SUBTYPES:
            raise ValueError(f'{source}: expected lossless WAV, got {info.format}/{info.subtype}')
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError(f'{source}: empty or invalid audio')
        prepared.append((record, sample.gt, source, info))

    output = new_directory(output_prefix).resolve()
    (output / 'audio').mkdir()
    status = output / 'generation.json'
    provenance = dict(status='building', source_manifest=str(data),
                      source_manifest_sha256=file_sha256(data),
                      condition='all_zero_waveform', count=len(records))
    status.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    original_records, silent_records, pairs = [], [], []
    try:
        for index, (record, gt, source, info) in enumerate(prepared):
            relative = f'audio/{index:06d}.wav'
            target = output / relative
            write_silence(target, info)
            verify_silence(target, info)
            original, silent = deepcopy(record), deepcopy(record)
            original['audio'] = str(source)
            silent['audio'] = relative
            # Preserve all annotations; embed recovered GT for standalone scoring.
            original['_gt'] = deepcopy(gt)
            silent['_gt'] = deepcopy(gt)
            original_records.append(original)
            silent_records.append(silent)
            pairs.append(dict(sample_id=record['id'], source_audio=str(source),
                              silent_audio=relative, frames=info.frames,
                              samplerate=info.samplerate, channels=info.channels,
                              subtype=info.subtype, duration_seconds=info.frames / info.samplerate,
                              source_sha256=file_sha256(source), silent_sha256=file_sha256(target)))
        for name, values in [('original.jsonl', original_records),
                             ('silent.jsonl', silent_records), ('pairs.jsonl', pairs)]:
            target = output / name
            temporary = target.with_suffix('.jsonl.tmp')
            with temporary.open('w', encoding='utf-8') as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            temporary.replace(target)
        provenance.update(status='complete', original_manifest_sha256=file_sha256(output / 'original.jsonl'),
                          silent_manifest_sha256=file_sha256(output / 'silent.jsonl'),
                          pairs_sha256=file_sha256(output / 'pairs.jsonl'))
    except Exception as error:
        provenance.update(status='failed', error=str(error))
        raise
    finally:
        status.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[Silent control] verified {len(records)} paired examples: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Evaluation JSONL with fixed questions')
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--output-prefix', default='silent_control')
    args = parser.parse_args()
    generate(args.data, args.audio_root, args.output_prefix)


if __name__ == '__main__':
    main()
