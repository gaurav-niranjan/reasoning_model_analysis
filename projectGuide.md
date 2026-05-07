# Thinking Models Project Guide

## Project Overview
This directory contains experiments with Qwen3-VL Thinking models. It runs resumable SB-Bench inference against a local vLLM OpenAI-compatible server, then uses an LLM-as-judge to classify whether a model's reasoning trace mentions specific background signals. Outputs are checkpointed JSONL files and consolidated Parquet results for analysis.

## Directory Structure
```
thinking_models/
  README.md
  pyproject.toml
  uv.lock
  vllm.sif
  projectGuide.md
  qwen/
    analyze_thoughts.ipynb
    compare_bg.ipynb
    judge_analysis.ipynb
    toy_run.ipynb
  vllm_files/
    check_server.py
    run_inference.py
    run_judge.py
    run_job.sh
    run_judge.sh
    test_script.sh
    logs/
    results/
```

## Module Breakdown
- README.md: High-level description of the thinking-model setup and evaluation goal.
- pyproject.toml: Python package metadata and runtime dependencies (PyTorch, transformers, diffusers, datasets, etc.).
- uv.lock: Dependency lockfile for reproducible installs.
- vllm.sif: Singularity container image used to launch the vLLM server on GPU nodes.
- qwen/*.ipynb: Analysis and visualization notebooks for reasoning traces, background comparisons, and judge outputs.
- vllm_files/check_server.py: Health check + smoke test against a running vLLM server (image + text request).
- vllm_files/run_inference.py: Asynchronous, resumable SB-Bench inference client. Builds prompts, loads images (original or inpainted), posts to vLLM, and writes JSONL checkpoints and Parquet.
- vllm_files/run_judge.py: LLM-as-judge pipeline. Reads inference Parquet, prompts a text-only vLLM model to classify background signals, and checkpoints results.
- vllm_files/run_job.sh: SLURM job to start the vLLM server in Singularity and run inference on SB-Bench.
- vllm_files/run_judge.sh: SLURM job to start a text-only vLLM judge server and run classification over inference outputs.
- vllm_files/test_script.sh: Interactive compute-node script for quick server smoke tests and a small inference run.
- vllm_files/logs/: SLURM and client logs.
- vllm_files/results/: Parquet and checkpoint outputs.

## Architecture & Connectivity
- vLLM server (Singularity) exposes OpenAI-style endpoints at `/health`, `/v1/models`, and `/v1/chat/completions`.
- `run_inference.py`:
  - Loads SB-Bench (HuggingFace `datasets`).
  - Chooses images from inpainted directories or the original dataset based on flags.
  - Sends image + prompt to the vLLM server, captures `reasoning` and final answer.
  - Writes an append-only JSONL checkpoint and consolidates into Parquet.
- `run_judge.py`:
  - Reads inference Parquet.
  - Sends thinking traces to a text-only judge model via the same vLLM API.
  - Parses JSON replies and writes checkpoints + Parquet outputs.

## Configuration & Execution
- Environment:
  - Python 3.10 is required (see [thinking_models/pyproject.toml](thinking_models/pyproject.toml)).
  - Activate `.venv` before running host-side clients.
  - vLLM server runs inside the Singularity image `vllm.sif`.
- SLURM inference:
  - Edit paths and experiment variables in [thinking_models/vllm_files/run_job.sh](thinking_models/vllm_files/run_job.sh) (model, inpainted dir, cache, output prefix).
  - Submit with `sbatch run_job.sh`.
  - Resume safely by re-submitting the same job; checkpoints are append-only.
- SLURM judge:
  - Edit judge inputs/outputs and signal in [thinking_models/vllm_files/run_judge.sh](thinking_models/vllm_files/run_judge.sh).
  - Submit with `sbatch run_judge.sh`.
- Interactive smoke test:
  - Run [thinking_models/vllm_files/test_script.sh](thinking_models/vllm_files/test_script.sh) inside a compute node to validate the server and pipeline.

## Typical Execution Path
1. Start vLLM server (Singularity) and verify it with `check_server.py`.
2. Run SB-Bench inference with `run_inference.py` (or via `run_job.sh`).
3. Convert checkpoints to Parquet and inspect outputs in the notebooks under `qwen/`.
4. Run `run_judge.py` (or `run_judge.sh`) to classify reasoning traces for background signals.
