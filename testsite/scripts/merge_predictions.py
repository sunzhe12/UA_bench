"""Compute exact aggregate classification metrics from multi-GPU shard outputs."""
import argparse
import json
from pathlib import Path

from testsite.core.protocol import validate_protocols
from testsite.core.scorer import Scorer
from testsite.core.integrity import validate_predictions


def aggregate(rows, manifest, protocols):
    if not rows:
        raise ValueError("no predictions found")
    manifest_hash = validate_predictions(rows, manifest)
    identity = validate_protocols(protocols, rows, manifest_hash)
    scorer = Scorer({"taxonomy": identity["taxonomy"]})
    gts = [x["gt"] for x in rows]
    l1p = [x["turn1_pred"] for x in rows]
    l2p = [x["turn2_pred"]["L2"] for x in rows]
    l3p = [x["turn2_pred"]["L3"] for x in rows]
    l1 = scorer.compute_level("L1", scorer.l1_keys, l1p, [x["L1"] for x in gts])
    l2 = scorer.compute_level("L2", scorer.l2_keys, l2p, [x["L2"] for x in gts])
    l3 = scorer.compute_level("L3", scorer.l3_keys, l3p, [x["L3"] for x in gts])
    l1_ok = [p == g["L1"] for p, g in zip(l1p, gts)]
    l2_ok = [p == g["L2"] for p, g in zip(l2p, gts)]
    l3_ok = [p == g["L3"] for p, g in zip(l3p, gts)]
    tiers = {}
    for row in rows:
        tier = str(row["turn2_pred"]["parse_tier"])
        tiers[tier] = tiers.get(tier, 0) + 1
    reasoning_rows = [x for x in rows if not x["cascade_skipped"]]
    reasoning = scorer.compute_reasoning_quality(
        [x["turn3_output"] for x in reasoning_rows],
        predicted_l3=[x["turn2_pred"]["L3"] for x in reasoning_rows],
    )
    reasoning.cascade_skipped = len(rows) - len(reasoning_rows)
    payload = {
        "manifest_sha256": manifest_hash, "run_signature": rows[0]["run_signature"],
        "integrity_verified": True,
        "total_samples": len(rows), "l1_accuracy": l1.accuracy,
        "l2_accuracy": l2.accuracy, "l3_accuracy": l3.accuracy,
        "l3_macro_f1": l3.macro_f1,
        "l2_given_l1": sum(a and b for a, b in zip(l1_ok, l2_ok)) / max(1, sum(l1_ok)),
        "l3_given_l2": sum(a and b for a, b in zip(l2_ok, l3_ok)) / max(1, sum(l2_ok)),
        "parse_tier_dist": tiers, "l3_per_class": l3.per_class,
        "l3_confusion_matrix": l3.confusion_matrix,
        "l3_confusion_labels": l3.confusion_labels,
        "reasoning": {
            "alignment_rate": reasoning.alignment_rate,
            "contradiction_rate": reasoning.contradiction_rate,
            "concept_confusion_rate": reasoning.concept_confusion_rate,
            "vague_rate": reasoning.vague_rate,
            "term_stacking_rate": reasoning.term_stacking_rate,
            "cascade_skipped": reasoning.cascade_skipped,
        },
    }
    return payload, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", nargs="+", required=True, help="All protocol_shard_NNN.json files for this run")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    payload, _ = aggregate(rows, args.manifest, args.protocol)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(rows)} predictions and scoring version; metrics -> {args.output}")


if __name__ == "__main__":
    main()
