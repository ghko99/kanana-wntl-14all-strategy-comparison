# Kanana-WNTL 14-all Strategy Comparison

This repository contains a standalone runner for comparing four inference strategies on the `14-all` AES dataset:

1. `hard_greedy`: greedy decoding hard digit scores
2. `soft_greedy`: greedy decoding with digit probability expected scores
3. `hard_sc`: hard self-consistency
4. `soft_sc`: soft weighted self-consistency

The comparison excludes `overall` QWK. It reports QWK for the 8 rubrics and their arithmetic mean.

## Required External Files

This repo includes `aes_dataset_mtl/test_14_all.jsonl`. It does not include model weights.

Prepare these paths on the machine where you run the script:

- `ADAPTER_DIR`: LoRA adapter directory, e.g. `kanana_wntl_20260407_002343`
- `BASE_MODEL`: fixed to `/shared/home/aif/hf_models/kanana` in `run_14_all_strategy_comparison.sh`
- `TEST_PATH`: defaults to `./aes_dataset_mtl/test_14_all.jsonl`

The dataset must contain either:

- `system`, `user`, `assistant`, or
- `instruction`, `output`

The assistant/output text must start with the 8 gold scores.

## Install

```bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

If your paths match the defaults:

```bash
./run_14_all_strategy_comparison.sh
```

For a different environment, pass paths through environment variables:

```bash
ADAPTER_DIR=/path/to/kanana_wntl_20260407_002343 \
DEVICE_ID=0 \
CHUNK_M=10 \
MAX_M=50 \
./run_14_all_strategy_comparison.sh
```

For a quick smoke test:

```bash
LIMIT=10 MAX_M=2 CHUNK_M=2 ./run_14_all_strategy_comparison.sh
```

## Outputs

Each run creates:

```text
strategy_comparison_results/<RUN_ID>/
```

Important files:

- `run.log`: full stdout/stderr log
- `run_config.json`: run configuration
- `ground_truth.jsonl`: gold rubric scores plus `grader_1_scores` and `grader_2_scores`
- `greedy_predictions.jsonl`: hard/soft greedy predictions
- `self_consistency_samples.jsonl`: all sampled SC generations plus grader scores
- `final_strategy_predictions.jsonl`: final predictions for all four strategies plus grader scores
- `strategy_summary.csv`: per-strategy scores
- `rubric_comparison.csv`: table-ready comparison
- `rubric_comparison.md`: Markdown table
- `curves.csv`, `curves.json`: average QWK curves by `m`
- `average_qwk_vs_m_with_greedy_baselines.png`: plot with greedy baselines and SC curves

## Notes

- `hard_sc` and `soft_sc` best `m` are selected by 8-rubric average QWK.
- `overall` QWK is intentionally not used.
- Increasing `CHUNK_M` can make sampling faster if GPU memory is sufficient.
- The default `MAX_SEQ_LENGTH=3072` matches the final comparison setup.
