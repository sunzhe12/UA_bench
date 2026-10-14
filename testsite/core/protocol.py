"""Persisted run identity and implementation checks for reproducible scoring."""
import hashlib
import json
from pathlib import Path
from .integrity import file_sha256

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    'core/parser.py', 'core/scorer.py', 'core/shared_terminology.py',
    'core/loader.py', 'core/dataset_record.py', 'core/integrity.py',
    'core/protocol.py', 'core/inference.py', 'eval/multi_turn.py',
    'scripts/run_eval.py', 'scripts/merge_predictions.py',
)


def implementation_hashes():
    return {'testsite/' + name: file_sha256(ROOT / name) for name in IMPLEMENTATION_FILES}


def identity_signature(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_protocols(paths, rows, manifest_hash):
    if not paths:
        raise ValueError('protocol files are required; missing run versions cannot be inferred')
    protocols = [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    current = implementation_hashes()
    expected_identity = protocols[0].get('run_identity')
    shards = set()
    for p in protocols:
        identity = p.get('run_identity')
        if not isinstance(identity, dict) or identity != expected_identity:
            raise ValueError('missing or inconsistent protocol run identity')
        if identity.get('implementation') != current:
            raise ValueError('implementation version mismatch: use the exact evaluation code/terminology version')
        if identity.get('manifest_sha256') != manifest_hash or p.get('manifest_sha256') != manifest_hash:
            raise ValueError('protocol manifest mismatch')
        if p.get('taxonomy') != identity.get('taxonomy') or not isinstance(identity.get('taxonomy'), dict):
            raise ValueError('protocol taxonomy mismatch')
        signature = identity_signature(identity)
        if p.get('run_signature') != signature or any(r.get('run_signature') != signature for r in rows):
            raise ValueError('protocol signature mismatch')
        index, count = p.get('shard_index'), p.get('num_shards')
        if type(index) is not int or type(count) is not int or not 0 <= index < count:
            raise ValueError('invalid protocol shard indices')
        if index in shards or count != protocols[0].get('num_shards'):
            raise ValueError('duplicate or inconsistent protocol shards')
        shards.add(index)
    if shards != set(range(protocols[0]['num_shards'])):
        raise ValueError('incomplete protocol shard coverage')
    if any(r.get('shard_index') not in shards or r.get('num_shards') != len(shards) for r in rows):
        raise ValueError('prediction/protocol shard mismatch')
    return expected_identity
