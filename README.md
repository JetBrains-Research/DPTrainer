# DPTrainer

[![CI](https://github.com/JetBrains-Research/DPTrainer/actions/workflows/ci.yaml/badge.svg)](https://github.com/JetBrains-Research/DPTrainer/actions/workflows/ci.yaml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

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

## What Is Differential Privacy?

Differential privacy (DP) is a mathematical framework that provides formal guarantees about the privacy of individuals in a dataset. A randomized algorithm *M* is **(ε, δ)-differentially private** if, for every pair of datasets *D* and *D′* that differ in a single record and for every set of possible outputs *S*:

> P[M(D) ∈ S] ≤ e^ε · P[M(D′) ∈ S] + δ

The parameter **ε** (epsilon) controls the privacy–utility trade-off: smaller ε means stronger privacy but typically lower model accuracy. **δ** bounds the probability that the guarantee fails. Together, (ε, δ) ensure that an adversary observing the algorithm's output cannot reliably determine whether any particular individual's data was included.

### DP-SGD: Differential Privacy for Deep Learning

Standard SGD computes gradients over a batch of training examples and applies their average to update model weights. **DP-SGD** modifies this process in two key ways:

1. **Per-sample gradient clipping** — each individual example's gradient is clipped to a fixed norm *C*, bounding the maximum influence any single record can have on the update.
2. **Noise injection** — calibrated Gaussian noise (scaled to *C*) is added to the aggregated gradient before the optimizer step, masking individual contributions.

A **privacy accountant** tracks the cumulative privacy cost (ε, δ) across all training steps using composition theorems (e.g., Rényi DP composition or the moments accountant), providing a formal end-to-end guarantee for the released model.

### Privacy Budget and Guarantee Failure

Differential privacy is not a binary property — it degrades gracefully as more computations touch the same data. Every DP-SGD step consumes a portion of the **privacy budget**, and the (ε, δ) guarantee applies to the *entire* training run, not to any single step in isolation.

**How composition works.** Each noisy gradient step is itself an (ε₁, δ₁)-DP mechanism. When *T* steps are composed, the total privacy cost grows with *T*. Naïve composition adds the per-step ε values linearly, but tighter analyses — such as the **moments accountant** (Abadi et al., 2016), **Rényi DP** composition (Mironov, 2017), and **privacy loss distributions** — grow closer to O(√T), making longer training runs feasible under a fixed budget.

**What ε actually bounds.** After training completes with a total budget of (ε, δ), any hypothesis test an adversary can run to decide whether a specific individual was in the training set has its advantage bounded: roughly, the log-likelihood ratio of any output under neighboring datasets is at most ε, except with probability δ. A smaller ε means the trained model "looks almost the same" regardless of any one person's participation.

**How the guarantee can fail (the role of δ).** The δ parameter represents the probability that the ε-bounded guarantee does not hold at all. With probability at most δ, the mechanism may produce an output that reveals an individual's participation with no bound on the likelihood ratio. In practice δ is set to a cryptographically small value — typically less than 1/*n* where *n* is the dataset size — so that this catastrophic failure event is negligibly rare.

**Budget exhaustion.** Once the accumulated privacy cost reaches the target (ε, δ), the guarantee covers everything the model has revealed so far. Continuing to train — or releasing intermediate checkpoints — without accounting for the additional cost would exceed the stated guarantee. This is why `DPTrainer` tracks the running budget and can automatically stop training when it is exhausted, ensuring the published (ε, δ) bound remains valid.

### Foundational Literature

- **Dwork, McSherry, Nissim & Smith (2006)** — [*Calibrating Noise to Sensitivity in Private Data Analysis*](https://link.springer.com/chapter/10.1007/11681878_14) — introduced the formal definition of (ε, δ)-differential privacy.
- **Dwork & Roth (2014)** — [*The Algorithmic Foundations of Differential Privacy*](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — comprehensive textbook covering the theory and core mechanisms of differential privacy.
- **Abadi et al. (2016)** — [*Deep Learning with Differential Privacy*](https://arxiv.org/abs/1607.00133) — introduced DP-SGD and the moments accountant for training deep neural networks with formal privacy guarantees.
- **Mironov (2017)** — [*Rényi Differential Privacy*](https://arxiv.org/abs/1702.07476) — proposed Rényi divergence-based privacy accounting, enabling tighter composition bounds.
- **Balle, Barthe & Gavin (2018)** — [*Privacy Amplification by Subsampling*](https://arxiv.org/abs/1807.01647) — formalized how random sub-sampling of data amplifies differential privacy guarantees.

## Why DPTrainer? Bridging Hugging Face Trainer and Opacus

[Opacus](https://opacus.ai/) is the standard PyTorch library for DP-SGD. It provides the core building blocks — per-sample gradient computation, DP optimizers, privacy accountants, and data loaders with Poisson sampling — but it is designed around a manual PyTorch training loop. [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) is the dominant high-level training API for Transformers, offering logging, checkpointing, evaluation, mixed-precision, and callback orchestration — but it has no awareness of differential privacy.

These two systems do not compose out of the box:

| Concern | Hugging Face Trainer | Opacus | DPTrainer |
|---|---|---|---|
| Optimizer creation | Internal; builds Adam/AdamW from `TrainingArguments` | Wraps any optimizer in a `DPOptimizer` | Intercepts `create_optimizer` to wrap the HF-created optimizer with Opacus's `DPOptimizer` |
| Gradient computation | Standard backprop (batch gradients) | Requires per-sample gradients via `GradSampleModule` | Wraps the model in Opacus's `GradSampleModule` controller before passing it to Trainer |
| Data loading | Standard `DataLoader` with fixed batches | `DPDataLoader` for Poisson-sampled batches | Overrides `get_train_dataloader` to return an Opacus `DPDataLoader` when Poisson sampling is enabled |
| Privacy accounting | Not supported | Manual — user must call the accountant each step | Automatically tracks (ε, δ) via a `DPCallback` hooked into the optimizer step |
| Noise calibration | Not supported | User computes and passes `noise_multiplier` | Automatically calibrates `noise_multiplier` from a target ε (or α, β) budget |
| Ghost clipping | Not supported | Provides `DPLossFastGradientClipping` primitive | Wraps the loss function with ghost clipping and warns if subclass overrides could bypass it |
| Checkpointing | Saves model/optimizer/scheduler state | No checkpoint integration | Saves and restores accountant state with HF checkpoints for correct budget tracking across restarts |
| Early stopping | Generic `EarlyStoppingCallback` | Not provided | Privacy-budget-aware early stopping that halts training when ε or β is exhausted |

In short, plugging Opacus into Hugging Face Trainer requires coordinated changes to model wrapping, optimizer creation, data loading, loss computation, checkpointing, and callback management. `DPTrainer` handles all of this so you can add differential privacy to any `Trainer`-based workflow — including third-party trainers like `DPOTrainer` or `Seq2SeqTrainer` via the `privatize_trainer` utility — without modifying their source code.

## Installation

```bash
pip install DPTrainer
```

Or with uv:

```bash
uv pip install DPTrainer
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from dptrainer import PrivacyArguments
from dptrainer.hugging_face import DPTrainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    per_sample_max_grad_norm=1.0,
)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=32,
)

trainer = DPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    privacy_args=privacy_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
```

## Privatizing Third-Party Trainers

`DPTrainer` works as a direct replacement for `Trainer`, but many Hugging Face workflows use specialized trainer subclasses — `Seq2SeqTrainer`, `DPOTrainer`, `SFTTrainer`, and others — that add task-specific logic on top of `Trainer`. Rewriting these classes to inherit from `DPTrainer` would be invasive and fragile.

The `privatize_trainer` utility solves this by patching any `Trainer`-based class at runtime, swapping `Trainer` for `DPTrainer` in its inheritance chain. The patched class retains all of its original behavior (custom loss functions, generation logic, reward computation) while gaining differential privacy:

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dptrainer import PrivacyArguments
from dptrainer.hugging_face.utils import privatize_trainer

privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    per_sample_max_grad_norm=1.0,
)

# One-line patch — Seq2SeqTrainer now trains with DP-SGD
privatize_trainer(Seq2SeqTrainer, default_privacy_args=privacy_args)

# Use Seq2SeqTrainer exactly as before — no other code changes needed
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

When ghost clipping is enabled, `privatize_trainer` automatically inspects the patched class's MRO and warns if any intermediate class overrides `compute_loss` or `training_step` in a way that could bypass DPTrainer's loss wrapping — so you get safety checks without manual auditing.

For a complete runnable example — including dataset preparation, model loading, and saving — see [Privatizing a Third-Party Trainer](docs/examples.md#privatizing-a-third-party-trainer) in the examples documentation.

## Documentation

For full documentation — including configuration reference, component details, noise calibration, and more — see the [docs](docs/index.md).

## License

Licensed under the Apache License, Version 2.0. See the repository LICENSE file and https://www.apache.org/licenses/LICENSE-2.0.
