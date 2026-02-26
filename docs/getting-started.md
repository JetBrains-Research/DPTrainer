# Getting Started

## Basic Usage with `DPTrainer`

`DPTrainer` is a drop-in replacement for Hugging Face's `Trainer` that adds differential privacy guarantees.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer

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
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Detach the model from the DP controller after training
model = trainer.detach_model()
model.save_pretrained("./my-private-model")
```

## Privatizing Any Trainer

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
