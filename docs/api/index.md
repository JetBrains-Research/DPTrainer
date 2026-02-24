# API Reference

Auto-generated documentation for all public classes and functions in `jbr-fed-dp-training`.

## Core

| Module | Description |
|---|---|
| [`PrivacyArguments`](privacy_arguments.md) | Dataclass for all privacy-related training parameters |

## Hugging Face Integration

| Module | Description |
|---|---|
| [`DPTrainer`](trainer.md) | Drop-in replacement for `transformers.Trainer` with DP support |
| [`DPCallback`](callback.md) | Trainer callback for privacy accounting and budget enforcement |
| [`privatize_trainer`](privatize_trainer.md) | Utility to patch any `Trainer` subclass for DP training |
| [`DataCollatorForCausalLM`](collator.md) | Data collator for causal LM with dynamic padding and label shifting |
| [`EarlyStoppingCallback`](early_stopping.md) | Checkpoint-aware early stopping callback |

## PyTorch

| Module | Description |
|---|---|
| [`pytorch`](pytorch.md) | Low-level PyTorch DP training loop |

## Utilities

| Module | Description |
|---|---|
| [`plots`](plots.md) | Privacy and loss visualization utilities |
| [`utils`](utils.md) | Helper utilities (model conversion, loss function setup) |
