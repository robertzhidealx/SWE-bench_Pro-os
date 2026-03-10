# SWE-bench Pro Harbor Adapter Parity Experiment

This repo is part of a broader parity experiment between the original SWE-bench Pro dataset and the version adapted to the Harbor framework. Specifically, it concerns producing results using the original dataset harness.

## Methodology

To effectively evaluate parity between the two datasets, Codex is set up for the original dataset in the exact same way it is set up in Harbor.

Since SWE-bench Pro contains a total of 731 large-scale software engineering tasks, to save cost, we randomly sample a diverse subset of 100 tasks for this experiment.

The parity experiment pipeline follows the exact steps required by SWE-bench Pro (see [README.md](README.md)). The `swe_bench_pro_eval.py` script is augmented to support remote execution on Daytona.

## Pipeline

1. Run Codex through the 100 sampled tasks:

```bash
uv run run_codex.py \
  --model gpt-5-mini-2025-08-07 \
  --instance_ids_file sampled_subset.txt \
  --output_dir results/swebenchpro-trial \
  --mode daytona \
  --max_concurrent 10 \
  --num_retries 3
```

2. Gather patches from the Codex outputs into one JSON file:

```bash
uv run helper_code/gather_patches.py \
  --directory results/swebenchpro-trial \
  --prefix swebenchpro-trial \
  --output results/swebenchpro-trial-patches.json
```

3. Evaluate the gathered patches against SWE-bench Pro tests:

```bash
uv run swe_bench_pro_eval.py \
  --raw_sample_path swebenchpro_raw.csv \
  --patch_path results/swebenchpro-trial-patches.json \
  --output_dir results/swebenchpro-trial-eval \
  --dockerhub_username jefzda \
  --scripts_dir run_scripts \
  --use_daytona \
  --num_workers 10
```
