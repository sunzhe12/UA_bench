"""Fail-fast manifest and prediction integrity checks, independent of metrics."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    records, seen = [], set()
    with open(path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {error.msg}') from error
            if not isinstance(item, dict):
                raise ValueError(f'{path}:{line_no}: expected a JSON object')
            sid = item.get('id')
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError(f'{path}:{line_no}: missing sample ID')
            if sid in seen:
                raise ValueError(f'{path}:{line_no}: duplicate sample ID {sid}')
            if not isinstance(item.get('audio'), str) or not item['audio'].strip():
                raise ValueError(f'{path}:{line_no}: {sid}: missing audio path')
            seen.add(sid)
            records.append(item)
    if not records:
        raise ValueError(f'{path}: empty manifest')
    return records


def validate_predictions(rows, manifest_path):
    """Require exact ID coverage and consistent inputs/run identity before scoring."""
    records = read_manifest(manifest_path)
    expected = {item['id']: item for item in records}
    seen = set()
    for row in rows:
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid:
            raise ValueError('prediction missing sample_id')
        if sid in seen:
            raise ValueError(f'duplicate prediction ID: {sid}')
        seen.add(sid)
    missing, extra = set(expected) - seen, seen - set(expected)
    if missing or extra:
        raise ValueError(f'prediction coverage mismatch: missing={sorted(missing)[:10]}, '
                         f'extra={sorted(extra)[:10]} (expected {len(expected)}, got {len(rows)})')
    signatures = {row.get('run_signature') for row in rows}
    if len(signatures) != 1 or not next(iter(signatures)):
        raise ValueError('missing or mixed run signatures; cannot verify one evaluation run')
    manifest_hash = file_sha256(manifest_path)
    for row in rows:
        sid = row['sample_id']
        item = expected[sid]
        if row.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'{sid}: prediction was not produced from this manifest')
        if item.get('_gt') and row.get('gt') != item['_gt']:
            raise ValueError(f'{sid}: ground truth differs from manifest')
        questions = [t.get('value') for t in item.get('conversations', []) if t.get('from') == 'human']
        if len(questions) != 3:
            raise ValueError(f'{sid}: manifest must contain three human questions')
        if row.get('turn1_prompt') != questions[0]:
            raise ValueError(f'{sid}: Turn 1 question differs from manifest')
        skipped = row.get('cascade_skipped')
        if not isinstance(skipped, bool) or skipped != (row['turn1_pred'] != row['gt']['L1']):
            raise ValueError(f'{sid}: inconsistent cascade state')
        for turn in (2, 3):
            if row.get(f'turn{turn}_prompt') != ('' if skipped else questions[turn - 1]):
                raise ValueError(f'{sid}: Turn {turn} question differs from manifest/gate')
        if row.get('qa_prompt_version') != item.get('qa_prompt_version', 'dataset_unversioned'):
            raise ValueError(f'{sid}: prompt version differs from manifest')
    return manifest_hash
