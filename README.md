# DPTrainer

[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![CI](https://github.com/JetBrains-Research/DPTrainer/actions/workflows/ci.yaml/badge.svg)](https://github.com/JetBrains-Research/DPTrainer/actions/workflows/ci.yaml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**DPTrainer** bridges [Opacus](https://opacus.ai/) and Hugging Face's [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) ecosystem — including the alignment trainers in [TRL](https://huggingface.co/docs/trl) (`DPOTrainer`, `SFTTrainer`, etc.) — so you can add DP-SGD to any `Trainer`-based workflow without rewriting training loops or modifying trainer source code.

## The Problem

Opacus provides the building blocks for DP-SGD (per-sample gradients, `DPOptimizer`, privacy accountants, Poisson-sampled data loaders), but it is designed around a manual PyTorch training loop. Hugging Face `Trainer` is the dominant high-level training API for Transformers — and TRL extends it with alignment-specific trainers — but neither has any awareness of differential privacy.

Plugging Opacus into a `Trainer`-based pipeline requires coordinated changes across model wrapping, optimizer creation, data loading, loss computation, checkpointing, and callback management. These concerns interact in non-obvious ways, and getting any one of them wrong silently breaks the privacy guarantee.

## What DPTrainer Handles

| Concern | Hugging Face Trainer / TRL | Opacus | DPTrainer |
|---|---|---|---|
| Optimizer creation | Internal; builds Adam/AdamW from `TrainingArguments` | Wraps any optimizer in `DPOptimizer` | Intercepts `create_optimizer` to wrap the HF-created optimizer with `DPOptimizer` |
| Gradient computation | Standard backprop (batch gradients) | Requires per-sample gradients via `GradSampleModule` | Wraps the model in `GradSampleModule` before passing it to Trainer |
| Data loading | Standard `DataLoader` with fixed batches | `DPDataLoader` for Poisson-sampled batches | Overrides `get_train_dataloader` to return `DPDataLoader` when Poisson sampling is enabled |
| Privacy accounting | Not supported | Manual — user calls the accountant each step | Automatically tracks (ε, δ) via a `DPCallback` hooked into the optimizer step |
| Noise calibration | Not supported | User computes and passes `noise_multiplier` | Automatically calibrates `noise_multiplier` from a target ε budget |
| Ghost clipping | Not supported | Provides `DPLossFastGradientClipping` primitive | Wraps the loss function with ghost clipping and warns if subclass overrides could bypass it |
| Checkpointing | Saves model/optimizer/scheduler state | No checkpoint integration | Saves and restores accountant state with HF checkpoints for correct budget tracking across restarts |
| Early stopping | Generic `EarlyStoppingCallback` | Not provided | Privacy-budget-aware early stopping that halts training when ε is exhausted |

## Installation

```bash
uv pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple DPTrainer
```

Or with pip:

```bash
pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple DPTrainer
```

## Quick Start — DPTrainer as a Drop-In Replacement

`DPTrainer` extends `transformers.Trainer`, so all standard training arguments, callbacks, checkpointing, and evaluation workflows work unchanged. Pass a `PrivacyArguments` instance to control the privacy budget:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from dptrainer import DPTrainer, PrivacyArguments

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

## Privatizing Any Trainer (TRL, Seq2Seq, etc.)

`DPTrainer` works as a direct replacement for `Trainer`, but many workflows use specialized subclasses — TRL's `DPOTrainer`, `SFTTrainer`, Hugging Face's `Seq2SeqTrainer`, and others — that add task-specific logic on top of `Trainer`. Rewriting them to inherit from `DPTrainer` would be invasive and fragile.

The `privatize_trainer` utility solves this by patching any `Trainer`-based class at runtime, swapping `Trainer` for `DPTrainer` in its inheritance chain. The patched class keeps all of its original behavior (custom loss, generation logic, reward computation) while gaining differential privacy:

```python
from trl import DPOTrainer
from transformers import TrainingArguments
from dptrainer import PrivacyArguments
from dptrainer import privatize_trainer

privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    per_sample_max_grad_norm=1.0,
)

# One-line patch — DPOTrainer now trains with DP-SGD
privatize_trainer(DPOTrainer)

# Use DPOTrainer exactly as before — no other code changes needed
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    privacy_args=privacy_args,
)
trainer.train()
```

> **Note:** `privatize_trainer` patches the trainer *class*, not a specific instance. It injects `DPTrainer` into the class hierarchy so that all future instances of that class train with DP-SGD. Use it once at import time or before constructing trainer instances.

> **Limitation:** `privatize_trainer` works only on **subclasses** of `transformers.Trainer` (e.g., `Seq2SeqTrainer`, `DPOTrainer`, `SFTTrainer`). It cannot be called on `Trainer` itself — doing so raises a `ValueError`. If you want DP training with the base `Trainer`, use `DPTrainer` directly as shown in the [Quick Start](#quick-start--dptrainer-as-a-drop-in-replacement) section above.

When ghost clipping is enabled, `privatize_trainer` automatically inspects the patched class's MRO and warns if any intermediate class overrides `compute_loss` or `training_step` in a way that could bypass DPTrainer's loss wrapping.

## PrivacyArguments Reference

| Parameter | Default | Description |
|---|---|---|
| `target_epsilon` | `None` | Target ε at end of training. Mutually exclusive with `noise_multiplier`. |
| `target_delta` | `None` | Target δ. Defaults to 1/N (dataset size). |
| `noise_multiplier` | `None` | Explicit noise multiplier. Mutually exclusive with `target_epsilon`. |
| `per_sample_max_grad_norm` | `0.5` | Max L2 norm for per-sample gradient clipping. |
| `clipping` | `"flat"` | Clipping strategy: `"flat"`, `"adaptive"` (AdaClip), or `"per_layer"`. |
| `poisson_sampling` | `True` | Use Poisson sub-sampling for privacy amplification. |
| `grad_sample_mode` | `"hooks"` | Opacus grad sample mode (`"hooks"` or `"ew"`). Use `"ew"` for ghost clipping. |
| `accountant` | `"rdp"` | Privacy accountant type (passed to Opacus). |

## Docs

For detailed documentation, please visit the [docs](docs/README.md).

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
