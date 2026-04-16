# Getting Started

## Basic Usage with `DPTrainer`

`DPTrainer` is a drop-in replacement for Hugging Face's `Trainer` that adds differential privacy guarantees.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from dptrainer import DPTrainer, PrivacyArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

privacy_args = PrivacyArguments(
    target_epsilon=8.0,       # Target privacy budget
    target_delta=1e-5,        # Target delta (defaults to 1/N if not set)
    per_sample_max_grad_norm=1.0,
    accountant="rdp",
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
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Detach the model from the DP controller after training
model = trainer.detach_model()
model.save_pretrained("./my-private-model")
```

## Privatizing Any Trainer

`DPTrainer` works as a direct replacement for `Trainer`, but many Hugging Face workflows rely on specialized trainer subclasses — `Seq2SeqTrainer`, `DPOTrainer`, `SFTTrainer`, and others — that add task-specific logic (custom loss functions, generation, reward computation) on top of `Trainer`. Rewriting these classes to inherit from `DPTrainer` would be invasive and fragile.

The `privatize_trainer` utility solves this by patching any `Trainer`-based class at runtime, swapping `Trainer` for `DPTrainer` in its method resolution order (MRO). The patched class retains all of its original behavior while gaining differential privacy:

```python
from trl import DPOTrainer
from dptrainer import PrivacyArguments
from dptrainer import privatize_trainer

privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    per_sample_max_grad_norm=1.0,
)

# One-line patch — DPOTrainer now trains with DP-SGD
privatize_trainer(DPOTrainer, default_privacy_args=privacy_args)

# Use DPOTrainer exactly as before — no other code changes needed
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

> **Limitation:** `privatize_trainer` can only be called on **subclasses** of `transformers.Trainer` — not on `Trainer` itself. Calling `privatize_trainer(Trainer)` raises a `ValueError` because `Trainer` does not have `Trainer` in its own inheritance chain, so there is nothing to swap. If you are using the base `Trainer` directly, use `DPTrainer` as a drop-in replacement instead (see [Basic Usage](#basic-usage-with-dptrainer) above).

When ghost clipping is enabled (`grad_sample_mode="ghost"` in `PrivacyArguments`), `privatize_trainer` automatically inspects the patched class's MRO and warns if any intermediate class overrides `compute_loss` or `training_step` in a way that could bypass `DPTrainer`'s loss wrapping — so you get safety checks without manual auditing.

For a complete runnable example — including dataset loading, preprocessing, and model saving — see [Privatizing a Third-Party Trainer](examples.md#privatizing-a-third-party-trainer) in the examples documentation.

## Saving and Loading

After training completes, detach the model from the DP controller before saving:

```python
model = trainer.detach_model()
model.save_pretrained("./my-private-model")
tokenizer.save_pretrained("./my-private-model")
```

The saved model is a standard Hugging Face model and can be loaded normally:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./my-private-model")
```

## Next Steps

- See [Configuration](configuration.md) for all `PrivacyArguments` options and noise calibration details.
- See [Examples](examples.md) for complete, runnable training scripts.
