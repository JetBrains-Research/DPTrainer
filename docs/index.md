# DPTrainer

Differential privacy training utilities for PyTorch and Hugging Face Transformers, powered by [Opacus](https://opacus.ai/).

## Overview

`DPTrainer` provides `DPTrainer` — a drop-in replacement for Hugging Face's `Trainer` that adds differential privacy (DP) guarantees via DP-SGD. It handles per-sample gradient clipping, noise injection, privacy budget accounting, and automatic noise calibration so you can fine-tune language models with formal privacy guarantees.

### Key Features

- **Drop-in Hugging Face integration** — `DPTrainer` extends `transformers.Trainer`, so all standard training arguments, callbacks, checkpointing, and evaluation workflows work out of the box.
- **Automatic noise calibration** — specify a target privacy budget (ε, δ) or error rates (α, β) and the noise multiplier is computed automatically.
- **Multiple privacy accountants** — supports Rényi DP (`rdp`) and the Connect-the-Dots accountant (`ctd`) from [riskcal](https://github.com/microsoft/riskcal) for tighter privacy analysis.
- **Gradient clipping strategies** — flat, adaptive (AdaClip), and per-layer clipping modes.
- **Poisson sampling** — optional Poisson sub-sampling for stronger privacy amplification.
- **Ghost clipping** — memory-efficient per-sample gradient computation via Opacus ghost clipping mode.
- **Privacy budget early stopping** — training automatically stops when the privacy budget (ε or β) is exhausted.
- **Checkpoint-aware accounting** — privacy accountant state is saved and restored with checkpoints for correct budget tracking across restarts.
- **`privatize_trainer` utility** — patch _any_ `Trainer`-based class (e.g., `DPOTrainer`, `Seq2SeqTrainer`) to use differential privacy without modifying its source code.
- **Patched components** — includes a checkpoint-aware `EarlyStoppingCallback` compatible with DP training.

## Installation

```bash
pip install DPTrainer
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install DPTrainer
```

## What's Next?

- [Getting Started](getting-started.md) — quick-start guide with code examples.
- [Configuration](configuration.md) — full reference for `PrivacyArguments` and noise calibration.
- [Examples](examples.md) — end-to-end training scripts.
- [API Reference](api/index.md). — auto-generated documentation for all public classes and functions.
