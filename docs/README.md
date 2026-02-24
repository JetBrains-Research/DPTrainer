# DPTrainer

Differential privacy training utilities for PyTorch and Hugging Face Transformers, powered by [Opacus](https://opacus.ai/).

## Overview

`DPTrainer` provides `DPTrainer` — a drop-in replacement for Hugging Face's `Trainer` that adds differential privacy (DP) guarantees via DP-SGD. It handles per-sample gradient clipping, noise injection, privacy budget accounting, and automatic noise calibration so you can fine-tune language models with formal privacy guarantees.

### Key Features

- **Drop-in Hugging Face integration** — `DPTrainer` extends `transformers.Trainer`, so all standard training arguments, callbacks, checkpointing, and evaluation workflows work out of the box.
- **Automatic noise calibration** — specify a target privacy budget (ε, δ) and the noise multiplier is computed automatically.
- **Privacy accounting via Opacus** — uses Rényi DP (`rdp`) accounting for end-to-end privacy tracking.
- **Gradient clipping strategies** — flat, adaptive (AdaClip), and per-layer clipping modes.
- **Poisson sampling** — optional Poisson sub-sampling for stronger privacy amplification.
- **Ghost clipping** — memory-efficient per-sample gradient computation via Opacus ghost clipping mode, with built-in safety guards that warn about incompatible method overrides and validate correct loss wrapping at runtime.
- **Privacy budget early stopping** — training automatically stops when the privacy budget (ε) is exhausted.
- **Checkpoint-aware accounting** — privacy accountant state is saved and restored with checkpoints for correct budget tracking across restarts.
- **`privatize_trainer` utility** — patch any `Trainer` **subclass** (e.g., `DPOTrainer`, `Seq2SeqTrainer`, `SFTTrainer`) to use differential privacy without modifying its source code. Note: cannot be used on `Trainer` itself — use `DPTrainer` directly for that case.
- **Single-GPU training** — designed for single-GPU training; distributed training (multi-GPU / multi-node) is not supported.
- **Patched components** — includes a checkpoint-aware `EarlyStoppingCallback` compatible with DP training.

## Installation

```bash
uv pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple DPTrainer
```

Or with pip:

```bash
pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple DPTrainer
```

## What's Next?

- [Getting Started](getting-started.md) — quick-start guide with code examples.
- [Configuration](configuration.md) — full reference for `PrivacyArguments` and noise calibration.
- [Examples](examples.md) — end-to-end training scripts.