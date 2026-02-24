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

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM

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
    data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
)

trainer.train()
```

## Documentation

For full documentation — including configuration reference, component details, noise calibration, and more — see the [docs](docs/README.md).

## License

Licensed under the Apache License, Version 2.0. See the repository LICENSE file and https://www.apache.org/licenses/LICENSE-2.0.
