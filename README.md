# UA-Bench

**Project page:** https://sunzhe12.github.io/ua-bench/ ·
**Paper:** https://arxiv.org/abs/2610.16942 ·
**Dataset:** https://huggingface.co/datasets/zhesun12/water

A three-level hierarchical classification benchmark for underwater acoustic signals. This repository contains the **evaluation framework**: it reads the released dataset, asks the three fixed questions embedded in each record, then parses and scores the answers.

| Level | Values |
|---|---|
| L1 | `active` / `passive` |
| L2 | `pulse` / `communication` / `ship_noise` |
| L3 | CW, LFM, HFM, 2FSK, 4FSK, BPSK, QPSK, OFDM, cargo, cruise, fishing, warship, underwater_target |

## Repository layout

```
README.md
testsite/
  core/       loading, answer parsing, scoring, model backends, statistics
  eval/       three-turn orchestration
  config/     taxonomy and default run settings
  scripts/    run, merge, silence generation, paired comparison, dataset download
  reporting/  Markdown report generation
  tests/      regression checks
```

## Dataset

The evaluation set is published on Hugging Face and is **not stored in this repository** (the archive is far above GitHub's 100 MB file limit):

- [`zhesun12/water`](https://huggingface.co/datasets/zhesun12/water) — `testset_export_v2.tar.gz`, 2600 records, 13 classes x 200, about 2.2 GB.

```bash
bash testsite/scripts/download_water_testset.sh                # extracts into ./testset_export
bash testsite/scripts/download_water_testset.sh /data/ua        # or choose a target directory
```

The script needs the `hf` CLI (`pip install huggingface_hub`). It verifies the archive size and SHA-256 and refuses entries containing absolute paths or `..`. `TESTSET_DIR`, `DATASET_REVISION`, `EXPECTED_SIZE` and `EXPECTED_SHA256` override the defaults; if the dataset on Hugging Face is replaced, update `revision` and the two checksums in the script. The result looks like this:

```
testset_export/
  sft_test_highquality.jsonl   # manifest
  audio/...                    # processed WAV (16 kHz)
```

## Installation

Python 3.10 or newer.

```bash
python -m pip install -r testsite/requirements.txt   # PyYAML / NumPy / SoundFile
```

PyTorch, Transformers and whatever each backend needs are installed according to that model's own documentation; this repository does not pin a GPU build.

## Running an evaluation

Run from the **directory that contains `testsite/`**.

The `audio` field in the manifest is a relative path, resolved as `<audio-root> / <audio field>`. A single missing audio file aborts the run.

Verify the environment with the mock backend first:

```bash
python -m testsite.scripts.run_eval --mock 26 --backend mock --output-dir results/mock_check
```

Single GPU (or a single process):

```bash
python -m testsite.scripts.run_eval \
  --data testset_export/sft_test_highquality.jsonl \
  --audio-root testset_export \
  --backend qwen25_omni \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/run_001
```

`--backend` also accepts `qwen2_audio`, `aero1_audio`, `voxtral_mini`, `voxtral_small`, `af_next`, `kimi_audio`, `minicpmo`, `gemma4` and `midashenglm`. Only `voxtral_*`, `qwen25_omni`, `af_next`, `gemma4` and `midashenglm` support true batching; the rest fall back to per-sample generation. Run `python -m testsite.scripts.run_eval --help` for the full option list.

### Sharding across GPUs

Every shard of one run must be given the **same `--run-id`**. The manifest is split deterministically by line number, so shards do not overlap:

```bash
RUN_ID=$(python -c 'import uuid; print(uuid.uuid4().hex)')
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m testsite.scripts.run_eval \
    --data testset_export/sft_test_highquality.jsonl \
    --audio-root testset_export \
    --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B \
    --num-shards 4 --shard-index $i --run-id "$RUN_ID" \
    --output-dir results/run_001 &
done
wait
```

### Merging

Concatenate the shard predictions into one file first, then merge; `--protocol` must list **all** protocol shards of that run:

```bash
cat results/run_001/predictions_shard_*.jsonl > results/run_001/predictions.jsonl

python -m testsite.scripts.merge_predictions \
  --input results/run_001/predictions.jsonl \
  --output results/run_001/metrics.json \
  --manifest testset_export/sft_test_highquality.jsonl \
  --protocol results/run_001/protocol_shard_*.json
```

Always point each run at a fresh output directory so that fixed-name prediction and protocol files are not overwritten.

## Output files

Each main run writes:

| File | Contents |
|---|---|
| `predictions_shard_NNN.jsonl` | Per sample: the questions actually sent, raw outputs, parse results, gating state |
| `protocol_shard_NNN.json` | Run signature, manifest hash, implementation hashes, taxonomy used |
| `eval_metrics_*.json` | L1/L2/L3 accuracy, L3 macro F1, L2\|L1, L3\|L2, per-class P/R/F1, confusion matrix |
| `eval_report_*.md` | The metrics above plus the five Turn 3 lexical diagnostics |

## Silence control (optional supplementary experiment)

Pairs the original audio with equal-duration, equal-rate silence under the same protocol and procedure.

```bash
# 1. Build the control; creates a fresh timestamped directory
python -m testsite.scripts.generate_silent_control \
  --data testset_export/sft_test_highquality.jsonl \
  --audio-root testset_export \
  --output-prefix silent_control

# 2. Run each condition once, same model and decoding settings
python -m testsite.scripts.run_eval --data CONTROL_DIR/original.jsonl --audio-root CONTROL_DIR \
  --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B --output-dir results/original
python -m testsite.scripts.run_eval --data CONTROL_DIR/silent.jsonl --audio-root CONTROL_DIR \
  --backend qwen25_omni --model-id Qwen/Qwen2.5-Omni-7B --output-dir results/silent

# 3. Verify the pairing and export the differences
python -m testsite.scripts.compare_silent_control \
  --control-dir CONTROL_DIR \
  --original-predictions results/original/predictions.jsonl \
  --silent-predictions results/silent/predictions.jsonl \
  --original-protocol results/original/protocol_shard_*.json \
  --silent-protocol results/silent/protocol_shard_*.json \
  --output-prefix results/comparison
```

Replace `CONTROL_DIR` with the directory printed by step 1. Differences are defined as **original minus silent** and are not assumed to be positive.

## Tests

```bash
python -m unittest discover -s testsite/tests
```

The test suite is the executable specification of the parsing, gating, scoring and integrity rules enforced by the framework.

## License

MIT — see [LICENSE](LICENSE).

## Notes

- The test set is a **diagnostic subset collected under favourable conditions** (the top 200 per class by channel gain and source SNR). It does not represent the full range of sea states or operational conditions.
- The exact answer format, cascade gating, metric definitions and integrity checks are implemented in `testsite/core/` and covered by `testsite/tests/`. Merging refuses a partial set of samples: a full-manifest evaluation cannot be substituted by a subset.
