"""Validate and compare paired original/silent runs using unchanged metrics."""
import argparse
import json
from pathlib import Path

from testsite.core.integrity import file_sha256, read_manifest
from testsite.core.statistics import (DEFAULT_REPLICATES, DEFAULT_SEED, summarise)
from testsite.scripts.generate_silent_control import new_directory
from testsite.scripts.merge_predictions import aggregate


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def compare(control_dir, original_predictions, silent_predictions,
            original_protocols, silent_protocols, output_prefix,
            replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED):
    control = Path(control_dir).resolve()
    generation = json.loads((control / 'generation.json').read_text(encoding='utf-8'))
    if generation.get('status') != 'complete':
        raise ValueError('silent generation is not complete')
    for name, key in [('original.jsonl', 'original_manifest_sha256'),
                      ('silent.jsonl', 'silent_manifest_sha256'), ('pairs.jsonl', 'pairs_sha256')]:
        if generation.get(key) != file_sha256(control / name):
            raise ValueError(f'generated manifest/hash mismatch: {name}')
    originals = {r['id']: r for r in read_manifest(control / 'original.jsonl')}
    silents = {r['id']: r for r in read_manifest(control / 'silent.jsonl')}
    pairs = read_rows(control / 'pairs.jsonl')
    ids = [p['sample_id'] for p in pairs]
    if (len(ids) != len(set(ids)) or set(ids) != set(originals)
            or set(ids) != set(silents) or generation.get('count') != len(ids)):
        raise ValueError('paired ID coverage mismatch')
    for pair in pairs:
        sid = pair['sample_id']
        a, b = originals[sid], silents[sid]
        if {k: v for k, v in a.items() if k != 'audio'} != {k: v for k, v in b.items() if k != 'audio'}:
            raise ValueError(f'{sid}: annotations/questions differ between conditions')
        for record, path_key, hash_key in [(a, 'source_audio', 'source_sha256'),
                                           (b, 'silent_audio', 'silent_sha256')]:
            audio = (control / record['audio']).resolve()
            if audio != (control / pair[path_key]).resolve() or file_sha256(audio) != pair[hash_key]:
                raise ValueError(f'{sid}: audio changed since control generation')

    rows_a, rows_b = read_rows(original_predictions), read_rows(silent_predictions)
    metrics_a, identity_a = aggregate(rows_a, control / 'original.jsonl', original_protocols)
    metrics_b, identity_b = aggregate(rows_b, control / 'silent.jsonl', silent_protocols)
    # Run IDs and manifest hashes must differ; model, decoding, taxonomy and code must agree.
    for key in ('model', 'taxonomy', 'implementation'):
        if identity_a[key] != identity_b[key]:
            raise ValueError(f'conditions use different {key}')
    keys = ('l1_accuracy', 'l2_accuracy', 'l3_accuracy', 'l3_macro_f1', 'l2_given_l1', 'l3_given_l2')
    deltas = {key: metrics_a[key] - metrics_b[key] for key in keys}
    def counts(rows):
        reached = [r for r in rows if not r['cascade_skipped']]
        return {'samples': len(rows), 'turn2_and_turn3_executed': len(reached),
                'turn1_invalid': sum(r.get('turn1_parse_status') == 'invalid_format' for r in rows),
                'turn2_invalid_among_executed': sum(r['turn2_pred'].get('parse_status') == 'invalid_format' for r in reached)}
    paired = []
    by_a, by_b = ({r['sample_id']: r for r in rows} for rows in (rows_a, rows_b))
    for sid in ids:
        a, b = by_a[sid], by_b[sid]
        paired.append({'sample_id': sid, 'gt': a['gt'],
                       'original': {'L1': a['turn1_pred'], 'L2': a['turn2_pred']['L2'], 'L3': a['turn2_pred']['L3']},
                       'silent': {'L1': b['turn1_pred'], 'L2': b['turn2_pred']['L2'], 'L3': b['turn2_pred']['L3']},
                       'original_l3_correct': a['turn2_pred']['L3'] == a['gt']['L3'],
                       'silent_l3_correct': b['turn2_pred']['L3'] == b['gt']['L3']})
    intervals = summarise(
        paired, [c['key'] for c in identity_a['taxonomy']['L3']['classes']],
        replicates=replicates, seed=seed)
    result = {'status': 'complete', 'comparison': 'original_minus_silent',
              'original': metrics_a, 'silent': metrics_b, 'delta': deltas,
              'confidence_intervals': intervals,
              'accuracy_delta_percentage_points': {k: 100*deltas[k] for k in keys if k.endswith('_accuracy')},
              'counts': {'original': counts(rows_a), 'silent': counts(rows_b)},
              'inputs': {str(Path(p).resolve()): file_sha256(p) for p in
                         [original_predictions, silent_predictions, *original_protocols, *silent_protocols]},
              'note': 'Invalid format is not refusal. Conditional and explanation metrics may use different eligible samples.'}
    output = new_directory(output_prefix)
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'paired_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in paired), encoding='utf-8')
    lines = ['# Original audio versus matched silence', '',
             'All values below use a 0–1 scale. Delta = original − silent.', '',
             '| Metric | Original | Silent | Delta |', '|---|---:|---:|---:|']
    lines += [f'| {k} | {metrics_a[k]:.6f} | {metrics_b[k]:.6f} | {deltas[k]:+.6f} |' for k in keys]
    lines += ['', result['note'], '', '## Confidence intervals (95%)', '',
              f"Resampling unit: {intervals['method']['resampling_unit']}.",
              f"Replicates: {intervals['method']['replicates']}, seed: {intervals['method']['seed']}, "
              f"RNG: {intervals['method']['rng']}.",
              f"Clusters: {intervals['sample_structure']['clusters']} for "
              f"{intervals['sample_structure']['samples']} samples "
              f"(max cluster size {intervals['sample_structure']['max_cluster_size']}).", '',
              '| Quantity | Condition | Estimate | 95% CI |', '|---|---|---:|---|']
    for group, label in (('accuracy', 'L3 accuracy'), ('macro_f1', 'Macro F1')):
        for condition, name in (('original', 'Original'), ('silent', 'Silent'),
                                ('delta', 'Delta (original − silent)')):
            cell = intervals[group][condition]
            lines.append(f"| {label} | {name} | {cell['estimate']:.6f} | "
                         f"[{cell['ci95'][0]:.6f}, {cell['ci95'][1]:.6f}] |")
    lines += ['', 'Cross-checks that do not depend on the random seed:', '',
              f"| Quantity | Condition | 95% CI |", '|---|---|---|',
              f"| L3 accuracy (Wilson, closed form) | Original | "
              f"[{intervals['accuracy']['original']['wilson_ci95'][0]:.6f}, "
              f"{intervals['accuracy']['original']['wilson_ci95'][1]:.6f}] |",
              f"| L3 accuracy (Wilson, closed form) | Silent | "
              f"[{intervals['accuracy']['silent']['wilson_ci95'][0]:.6f}, "
              f"{intervals['accuracy']['silent']['wilson_ci95'][1]:.6f}] |",
              f"| L3 accuracy delta (paired, analytic) | Delta | "
              f"[{intervals['accuracy']['delta']['analytic_ci95'][0]:.6f}, "
              f"{intervals['accuracy']['delta']['analytic_ci95'][1]:.6f}] |", '',
              f"Discordant pairs (McNemar): original-only correct "
              f"{intervals['accuracy']['delta']['discordant_pairs']['original_only']}, "
              f"silent-only correct {intervals['accuracy']['delta']['discordant_pairs']['silent_only']}.",
              '', intervals['note'], '',
              'Execution/invalid-format counts:', '',
              '```json', json.dumps(result['counts'], indent=2), '```']
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--original-predictions', required=True)
    parser.add_argument('--silent-predictions', required=True)
    parser.add_argument('--original-protocol', nargs='+', required=True)
    parser.add_argument('--silent-protocol', nargs='+', required=True)
    parser.add_argument('--output-prefix', default='silent_comparison')
    parser.add_argument('--replicates', type=int, default=DEFAULT_REPLICATES,
                        help='Cluster bootstrap replicates for interval estimates')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Bootstrap seed (recorded in the output for auditability)')
    args = parser.parse_args()
    print(compare(args.control_dir, args.original_predictions, args.silent_predictions,
                  args.original_protocol, args.silent_protocol, args.output_prefix,
                  replicates=args.replicates, seed=args.seed))


if __name__ == '__main__':
    main()
