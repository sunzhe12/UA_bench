#!/usr/bin/env python
"""Run the multi-turn dialogue evaluation pipeline.

Usage:
    python -m testsite.scripts.run_eval
    python -m testsite.scripts.run_eval --data sft_test.jsonl
    python -m testsite.scripts.run_eval --mock 50
"""

import sys
import argparse
import json
import hashlib
import uuid
import os
from pathlib import Path

# libgomp requires a positive integer. Some rented GPU images export this as an
# empty or malformed value, producing warnings before model inference starts.
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from testsite.config import load_config
from testsite.core.loader import DataLoader
from testsite.core.integrity import file_sha256
from testsite.core.protocol import implementation_hashes, identity_signature
from testsite.core.inference import ModelInference
from testsite.core.scorer import Scorer
from testsite.eval.multi_turn import MultiTurnEvaluator
from testsite.reporting.report import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Underwater Acoustic LLM — Hierarchical Classification Evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path (default: config/eval_config.yaml)")
    parser.add_argument("--data", type=str, default=None,
                        help="Test data JSONL path (default: mock mode)")
    parser.add_argument("--audio-root", type=str, default="processed_audio",
                        help="Processed audio root directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Load only the first N real samples (smoke testing)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split the manifest deterministically across N workers")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based worker index used with --num-shards")
    parser.add_argument("--mock", type=int, default=None,
                        help="Generate N mock samples (default: 50 when no data)")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["mock", "qwen2_audio", "aero1_audio",
                                 "voxtral_mini", "voxtral_small", "af_next", "kimi_audio",
                                 "qwen25_omni", "minicpmo", "gemma4",
                                 "midashenglm"],
                        help="Override model.backend in config")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model.model_id in config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="True generation batch size where the backend supports it")
    parser.add_argument("--device", type=str, default=None,
                        help="Device/device_map override, e.g. auto or cuda:0")
    parser.add_argument("--attn-implementation", type=str, default=None,
                        choices=["flash_attention_2", "sdpa"],
                        help="Attention backend override")
    parser.add_argument("--run-id", default=None, help="Shared identity for shards of one run")
    args = parser.parse_args()
    args.run_id = args.run_id or uuid.uuid4().hex
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")

    # ============================================================
    # 1. Load configuration
    # ============================================================
    print("=" * 60)
    print("  Underwater Acoustic LLM — Hierarchical Classification")
    print("=" * 60)

    config = load_config(args.config)
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.model_id:
        config["model"]["model_id"] = args.model_id
    if args.batch_size:
        config["model"]["batch_size"] = args.batch_size
    if args.device:
        config["model"]["device"] = args.device
    if args.attn_implementation:
        config["model"]["attn_implementation"] = args.attn_implementation

    # Auto-create subdirectory per model to avoid overwriting results
    backend_name = config["model"]["backend"]
    if args.output_dir == "eval_results" and backend_name != "mock":
        args.output_dir = f"eval_results/{backend_name}/dataset_questions"

    # ============================================================
    # 2. Load data
    # ============================================================
    loader = DataLoader(config)

    if args.data:
        print(f"\nLoading test data: {args.data}")
        samples = loader.load(jsonl_path=args.data, audio_root=args.audio_root,
                              limit=args.limit, shard_index=args.shard_index,
                              num_shards=args.num_shards)
    else:
        n_mock = args.mock if args.mock is not None else 50
        print(f"\nNo data file, using mock mode ({n_mock} samples)")
        samples = _mock_samples(config, n_mock)

    if not samples:
        print("Error: no samples to evaluate")
        return 1

    print(f"  Loaded {len(samples)} samples")

    # ============================================================
    # 3. Initialise
    # ============================================================
    manifest_sha256 = file_sha256(args.data) if args.data else "mock"
    identity = {"run_id": args.run_id, "manifest_sha256": manifest_sha256, "model": config["model"],
                "taxonomy": config["taxonomy"],
                "implementation": implementation_hashes()}
    # Worker batch/device choices must not prevent compatible shard merging.
    identity["model"] = {k: v for k, v in identity["model"].items() if k not in ("device", "batch_size")}
    run_signature = identity_signature(identity)
    inference = ModelInference(config)

    # ============================================================
    # 4. Multi-turn dialogue evaluation
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  Multi-Turn Evaluation")
    print(f"{'=' * 60}")

    multi_eval = MultiTurnEvaluator(config, inference)
    hier_metrics, reasoning, turn_results = multi_eval.evaluate(samples)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    protocol_path = output_path / f"protocol_shard_{args.shard_index:03d}.json"
    protocol_path.write_text(json.dumps({
        "run_signature": run_signature,
        "manifest_sha256": manifest_sha256,
        "run_identity": identity,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "question_source": "dataset.conversations.human",
        "qa_prompt_versions": sorted({s.qa_prompt_version for s in samples}),
        "taxonomy": config["taxonomy"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = output_path / f"predictions_shard_{args.shard_index:03d}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fout:
        for r in turn_results:
            fout.write(json.dumps({
                "run_signature": run_signature,
                "manifest_sha256": manifest_sha256,
                "shard_index": args.shard_index, "num_shards": args.num_shards,
                "qa_prompt_version": r.sample.qa_prompt_version,
                "question_source": "dataset.conversations.human",
                "turn1_prompt": r.turn1_prompt,
                "turn2_prompt": r.turn2_prompt,
                "turn3_prompt": r.turn3_prompt,
                "sample_id": r.sample.sample_id, "gt": r.sample.gt,
                "metadata": r.sample.metadata,
                "turn1_pred": r.turn1_pred.L1,
                "turn1_output": r.turn1_output,
                "turn1_parse_status": r.turn1_pred.parse_status,
                "turn2_pred": {"L1": r.turn2_pred.L1, "L2": r.turn2_pred.L2,
                               "L3": r.turn2_pred.L3, "parse_tier": r.turn2_pred.parse_tier,
                               "parse_status": r.turn2_pred.parse_status},
                "turn2_output": r.turn2_output, "turn3_output": r.turn3_output,
                "cascade_skipped": r.cascade_skipped,
            }, ensure_ascii=False) + "\n")

    print(f"  T1 L1 Accuracy:     {hier_metrics.l1.accuracy:.2%}")
    print(f"  T2 L2 Accuracy:     {hier_metrics.l2.accuracy:.2%}")
    print(f"  T2 L3 Accuracy:     {hier_metrics.l3.accuracy:.2%}")
    print(f"  L3 Macro F1:        {hier_metrics.l3.macro_f1:.4f}")
    print(f"  L2|L1:              {hier_metrics.l2_given_l1:.2%}")
    print(f"  L3|L2:              {hier_metrics.l3_given_l2:.2%}")
    print()
    print(f"\n  T3 Alignment:       {reasoning.alignment_rate:.2%}")
    print(f"  T3 Contradiction:   {reasoning.contradiction_rate:.2%}")
    print(f"  Concept Confusion:   {reasoning.concept_confusion_rate:.2%}")
    print(f"  Vague / Generic:     {reasoning.vague_rate:.2%}")
    print(f"  Term Stacking:       {reasoning.term_stacking_rate:.2%}")

    # ============================================================
    # 5. Generate report
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  Generating Report")
    print(f"{'=' * 60}")

    report_gen = ReportGenerator(output_dir=args.output_dir)
    report_text = report_gen.generate(
        hier_metrics,
        model_name="Underwater Acoustic LLM",
        reasoning=reasoning,
    )

    # ============================================================
    # 6. Summary
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Complete")
    print(f"  L3 Accuracy: {hier_metrics.l3.accuracy:.2%}")
    print(f"  Report:      {args.output_dir}/")
    print(f"{'=' * 60}")

    return 0


def _mock_samples(config, n):
    """Generate mock samples using MockModel (for testing the framework)."""
    from testsite.core.inference import MockModel
    from testsite.core.loader import EvalSample
    import random
    rng = random.Random(42)

    l3_classes = [
        ("CW", "pulse", "active"), ("LFM", "pulse", "active"), ("HFM", "pulse", "active"),
        ("2FSK", "communication", "active"), ("4FSK", "communication", "active"),
        ("BPSK", "communication", "active"), ("QPSK", "communication", "active"),
        ("OFDM", "communication", "active"),
        ("cargo", "ship_noise", "passive"), ("cruise", "ship_noise", "passive"),
        ("fishing", "ship_noise", "passive"), ("warship", "ship_noise", "passive"),
        ("underwater_target", "ship_noise", "passive"),
    ]

    samples = []
    for i in range(n):
        l3, l2, l1 = rng.choice(l3_classes)
        sample_id = f"mock_{l3}_{i:06d}"
        audio_path = f"mock/audio/{l3}/{sample_id}.wav"
        # Fixtures supply their questions explicitly, just like loaded records.
        prompts = config["prompts"]["three_turn"]
        branch = "turn2_active" if l1 == "active" else "turn2_passive"
        questions = (prompts["turn1"][0], prompts[branch][0].replace(
            "{L1}", "actively transmitted" if l1 == "active" else "source-radiated"),
            prompts["turn3"])
        samples.append(EvalSample(
            sample_id=sample_id,
            audio_path=audio_path,
            questions=questions,
            qa_prompt_version=config.get("qa_prompt_version", "mock"),
            gt={"L1": l1, "L2": l2, "L3": l3},
            metadata={
                "snr_db": rng.uniform(-15, 25),
                "ssp_complexity": rng.uniform(0, 10),
            },
        ))
    return samples


if __name__ == "__main__":
    sys.exit(main())
