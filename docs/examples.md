# Examples

Complete, runnable example scripts are available in the [`examples/`](https://github.com/JetBrains-Research/DPTrainer/tree/main/examples) directory.

## End-to-End Causal LM Fine-Tuning

Fine-tune GPT-2 on Wikitext-2 with differential privacy:

```python
"""docs/examples/causal_lm_finetuning.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dptrainer import DPTrainer, PrivacyArguments


def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)

    # Configure privacy
    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
        accountant="rdp",
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir="./output/causal-lm-dp",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    # Train with differential privacy
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        privacy_args=privacy_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save the model
    model = trainer.detach_model()
    model.save_pretrained("./output/causal-lm-dp/final")
    tokenizer.save_pretrained("./output/causal-lm-dp/final")


if __name__ == "__main__":
    main()
```

## Privatizing a Third-Party Trainer

Use `privatize_trainer` to add DP to any `Trainer`-based class, such as `Seq2SeqTrainer`:

> **Note:** `privatize_trainer` only works on `Trainer` subclasses — it cannot be applied to `transformers.Trainer` itself. If you are using the base `Trainer`, use `DPTrainer` directly instead. See the [Causal LM Fine-Tuning](#end-to-end-causal-lm-fine-tuning) example above.

```python
"""docs/examples/privatize_seq2seq.py"""

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dptrainer import PrivacyArguments
from dptrainer import privatize_trainer


def main():
    # Load model and tokenizer
    model_name = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and preprocess dataset
    raw_dataset = load_dataset("imdb", split="train[:16]")

    def preprocess_function(examples):
        inputs = ["classify: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        # Convert labels to strings for T5 seq2seq generation
        labels = tokenizer([str(label) for label in examples["label"]], max_length=8, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_dataset.map(preprocess_function, batched=True)

    # Configure privacy
    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
    )

    # Patch Seq2SeqTrainer to use DPTrainer under the hood
    privatize_trainer(Seq2SeqTrainer)

    # Configure training
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output/seq2seq-dp",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    # Train with differential privacy
    # Seq2SeqTrainer now trains with differential privacy automatically
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        privacy_args=privacy_args,
    )

    trainer.train()

    # Save the model
    model.save_pretrained("./output/seq2seq-dp/final")
    tokenizer.save_pretrained("./output/seq2seq-dp/final")


if __name__ == "__main__":
    main()
```

## Using a Stricter Epsilon Budget

Train with a stricter epsilon target for stronger privacy:

```python
"""docs/examples/strict_epsilon.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dptrainer import DPTrainer, PrivacyArguments


def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)

    # Configure privacy with a strict epsilon budget
    privacy_args = PrivacyArguments(
        target_epsilon=4.0,
        target_delta=1e-5,
        per_sample_max_grad_norm=1.0,
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir="./output/strict-epsilon-dp",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    # Train with differential privacy
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        privacy_args=privacy_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save the model
    model = trainer.detach_model()
    model.save_pretrained("./output/strict-epsilon-dp/final")
    tokenizer.save_pretrained("./output/strict-epsilon-dp/final")


if __name__ == "__main__":
    main()
```

## Adaptive Clipping

Use adaptive clipping (AdaClip) to dynamically adjust the clipping bound during training:

```python
"""docs/examples/adaptive_clipping.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dptrainer import DPTrainer, PrivacyArguments


def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)

    # Configure privacy with adaptive clipping
    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
        clipping="adaptive",
        target_unclipped_quantile=0.5,
        clipbound_learning_rate=0.2,
        min_clipbound=0.05,
        max_clipbound=100.0,
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir="./output/adaptive-dp",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    # Train with differential privacy
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        privacy_args=privacy_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save the model
    model = trainer.detach_model()
    model.save_pretrained("./output/adaptive-dp/final")
    tokenizer.save_pretrained("./output/adaptive-dp/final")


if __name__ == "__main__":
    main()
```

## Ghost Clipping

Ghost clipping is a memory-efficient alternative to the default hook-based per-sample gradient computation.
Instead of materializing full per-sample gradients, it computes per-sample gradient *norms* in a first
backward pass and then re-scales a standard (aggregated) gradient in a second pass. This can dramatically
reduce GPU memory usage — especially for large models — while producing mathematically equivalent updates.


Ghost clipping requires the model to expose a `loss_function` attribute — `DPTrainer` wraps it
with `DPLossFastGradientClipping` to compute per-sample gradient norms during the forward pass.
This means ghost clipping **cannot be used with trainer subclasses that override `compute_loss`
or `training_step`** (e.g., `DPOTrainer`, `SFTTrainer`), because those overrides bypass the
wrapped loss function and break privacy gradient computation. When you call `privatize_trainer`
with `grad_sample_mode="ghost"`, it inspects the class hierarchy and emits a warning if such
overrides are detected; at runtime, `DPCallback` raises a `RuntimeError` if the wrapped loss
has been replaced. Additionally, **adaptive clipping is not supported** with ghost clipping and
will silently fall back to flat clipping.

Enable ghost clipping by setting `grad_sample_mode="ghost"` in `PrivacyArguments`:

```python
"""docs/examples/ghost_clipping.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from dptrainer import DPTrainer, PrivacyArguments


def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)

    # Configure privacy with ghost clipping
    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
        grad_sample_mode="ghost",  # Use ghost clipping for lower memory usage
    )

    # Configure training
    training_args = TrainingArguments(
        output_dir="./output/ghost-dp",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    # Train with differential privacy using ghost clipping
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        privacy_args=privacy_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save the model
    model = trainer.detach_model()
    model.save_pretrained("./output/ghost-dp/final")
    tokenizer.save_pretrained("./output/ghost-dp/final")


if __name__ == "__main__":
    main()
```
