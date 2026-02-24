# jbr-fed-dp-training

Differential privacy training utilities for PyTorch and Hugging Face Transformers, powered by [Opacus](https://opacus.ai/).

## Overview

`jbr-fed-dp-training` provides `DPTrainer` — a drop-in replacement for Hugging Face's `Trainer` that adds differential privacy (DP) guarantees via DP-SGD. It handles per-sample gradient clipping, noise injection, privacy budget accounting, and automatic noise calibration so you can fine-tune language models with formal privacy guarantees.

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
- **Patched components** — includes `DataCollatorForCausalLM` and a checkpoint-aware `EarlyStoppingCallback` compatible with DP training.

## Installation

```bash
pip install jbr-fed-dp-training
```

Or with uv:

```bash
uv pip install jbr-fed-dp-training
```

During monorepo development, the package is linked as an editable dependency via `[tool.uv.sources]` in `pyproject.toml`.

## Quick Start

### Basic Usage with `DPTrainer`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

privacy_args = PrivacyArguments(
    target_epsilon=8.0,       # Target privacy budget
    target_delta=1e-5,        # Target delta (defaults to 1/N if not set)
    per_sample_max_grad_norm=1.0,
    accountant="rdp",         # "rdp" or "ctd"
)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = DPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    privacy_args=privacy_args,
    data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
)

trainer.train()

# Detach the model from the DP controller after training
model = trainer.detach_model()
model.save_pretrained("./my-private-model")
```

### Privatizing Any Trainer (e.g., DPOTrainer)

Use `privatize_trainer` to add differential privacy to any `Trainer` subclass without changing its code:

```python
from trl import DPOTrainer
from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face.utils import privatize_trainer

privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    per_sample_max_grad_norm=1.0,
)

# Patch DPOTrainer to inherit from DPTrainer instead of Trainer
privatize_trainer(DPOTrainer, default_privacy_args=privacy_args)

# Now use DPOTrainer as usual — it trains with DP automatically
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Configuration

### `PrivacyArguments`

All privacy-related parameters are configured through the `PrivacyArguments` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `accountant` | `"rdp"` | Privacy accountant mechanism (`"rdp"` or `"ctd"`) |
| `noise_multiplier` | `None` | Noise multiplier σ for DP-SGD. Auto-calculated if `None` |
| `target_epsilon` | `None` | Target ε at end of training (used to auto-calculate σ) |
| `target_delta` | `None` | Target δ (defaults to `1/N` where N is the dataset size) |
| `target_alpha` | `0.0001` | Target FPR for β search (used with `ctd` accountant) |
| `target_beta` | `None` | Target FNR (alternative to ε for noise calibration) |
| `per_sample_max_grad_norm` | `0.5` | Maximum gradient norm for per-sample clipping |
| `clipping` | `"flat"` | Clipping strategy: `"flat"`, `"adaptive"`, or `"per_layer"` |
| `poisson_sampling` | `True` | Use Poisson sub-sampling for privacy amplification |
| `grad_sample_mode` | `"hooks"` | Opacus grad sample mode: `"hooks"` or `"ghost"` |

#### Adaptive Clipping (AdaClip) Parameters

When using `clipping="adaptive"`, the following additional parameters are available:

| Parameter | Default | Description |
|---|---|---|
| `min_clipbound` | `0.05` | Minimum clip bound |
| `max_clipbound` | `1e8` | Maximum clip bound |
| `clipbound_learning_rate` | `0.2` | Learning rate for clip bound adaptation |
| `target_unclipped_quantile` | `0.5` | Target fraction of unclipped samples |
| `unclipped_num_std` | `1.0` | Standard deviation of unclipped number noise |

### Noise Calibration

The noise multiplier is determined automatically based on the privacy target you provide. The priority is:

1. **Explicit `noise_multiplier`** — used directly if provided.
2. **`target_epsilon`** — noise is calibrated to achieve the target (ε, δ)-DP using the specified accountant.
3. **`target_beta`** — noise is calibrated to achieve the target (α, β) error rates using [riskcal](https://github.com/microsoft/riskcal).
4. **None specified** — defaults to `noise_multiplier=0.0` (no noise, no privacy).

### Privacy Budget Early Stopping

`DPTrainer` automatically monitors the privacy budget during training. If a `target_epsilon` or `target_beta` is set, training will stop early when the budget is exhausted. This is handled by the built-in `DPCallback`.

Privacy metrics (`privacy_epsilon`, `privacy_beta`, `privacy_advantage`) are logged alongside standard training metrics during evaluation.

## Components

### `DPTrainer`

The main trainer class. Extends `transformers.Trainer` with:

- Opacus model wrapping via `GradSampleModule` controller
- DP optimizer wrapping (flat, adaptive, or per-layer clipping)
- Privacy-aware data loading with optional Poisson sampling and batch memory management
- Automatic privacy metric computation during evaluation
- Ghost clipping support for memory-efficient training

**Key methods:**

- `detach_model()` — cleanly detaches the model from the DP controller after training, returning the underlying `nn.Module` for saving or inference.

### `DPCallback`

A `TrainerCallback` that integrates Opacus privacy accounting into the Hugging Face training loop. It:

- Tracks privacy budget consumption at each optimizer step
- Reports privacy metrics (ε, β, advantage) during evaluation
- Stops training when the privacy budget is exceeded
- Saves and restores accountant state with checkpoints

### `privatize_trainer`

A utility function that dynamically patches any `Trainer` subclass to inherit from `DPTrainer`. This is useful for adding DP to third-party trainers (e.g., `DPOTrainer` from TRL) without modifying their source.

### `DataCollatorForCausalLM`

A data collator for causal language modeling that dynamically pads inputs and creates shifted labels. Filters out empty sequences and adds `position_ids` for compatibility with DP training.

### `EarlyStoppingCallback`

A patched version of Hugging Face's `EarlyStoppingCallback` that correctly handles checkpoint restoration — it checks at `on_train_begin` whether the patience counter was already exceeded before resuming training.

## Requirements

- Python ≥ 3.11, < 3.13
- PyTorch ≥ 2.6
- Transformers ≥ 4.54.1
- Opacus 1.5.4.post10
- riskcal 1.2.0.dev1

## Development

```bash
# Sync dependencies
uv sync

# Run tests
uv run pytest -q
```

## Layout

```
packages/dp-training/
├── pyproject.toml
├── README.md
├── src/
│   └── jbr/fed/dp_training/
│       ├── privacy_arguments.py     # PrivacyArguments dataclass
│       ├── hugging_face/
│       │   ├── trainer.py           # DPTrainer
│       │   ├── callback.py          # DPCallback
│       │   ├── utils/               # privatize_trainer utility
│       │   └── patched/             # DataCollatorForCausalLM, EarlyStoppingCallback
│       ├── pytorch/                 # PyTorch DP training loop
│       ├── plots.py                 # Privacy visualization utilities
│       └── utils/                   # Helper utilities
└── tests/
```

## Releases and Versioning

- Version is derived from Git tags (`0.0.0` in `pyproject.toml` for local builds).
- Tag format: `dp-training-vX.Y.Z` (e.g., `dp-training-v0.1.0`).

```bash
git tag dp-training-v0.1.0
git push origin dp-training-v0.1.0
```

## License

Licensed under the Apache License, Version 2.0. See the repository LICENSE file and https://www.apache.org/licenses/LICENSE-2.0.
