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
| [`EarlyStoppingCallback`](early_stopping.md) | Checkpoint-aware early stopping callback |

## Utilities

| Module | Description |
|---|---|
| [`utils`](utils.md) | Helper utilities (model conversion, loss function setup) |
