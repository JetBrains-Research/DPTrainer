# Examples

Complete, runnable example scripts are available in the [`examples/`](https://github.com/JetBrains-Research/DPTrainer/tree/main/examples) directory.

## End-to-End Causal LM Fine-Tuning

Fine-tune GPT-2 on Wikitext-2 with differential privacy:

```python
"""examples/causal_lm_finetuning.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM


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
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
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

```python
"""examples/privatize_seq2seq.py"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face.utils import privatize_trainer


def main():
    model_name = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
    )

    # Patch Seq2SeqTrainer to use DPTrainer under the hood
    privatize_trainer(Seq2SeqTrainer, default_privacy_args=privacy_args)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./output/seq2seq-dp",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        report_to="none",
    )

    # Seq2SeqTrainer now trains with differential privacy automatically
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # your dataset here
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

## Using the CTD Accountant with Beta Targets

Train with the Connect-the-Dots accountant using error rate targets instead of epsilon:

```python
"""examples/ctd_accountant.py"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)

    # Configure privacy with CTD accountant and beta target
    privacy_args = PrivacyArguments(
        target_alpha=0.0001,   # False positive rate
        target_beta=0.1,       # False negative rate — training stops when exceeded
        per_sample_max_grad_norm=1.0,
        accountant="ctd",
    )

    training_args = TrainingArguments(
        output_dir="./output/ctd-dp",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        privacy_args=privacy_args,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
    )

    trainer.train()

    model = trainer.detach_model()
    model.save_pretrained("./output/ctd-dp/final")


if __name__ == "__main__":
    main()
```

## Adaptive Clipping

Use adaptive clipping (AdaClip) to dynamically adjust the clipping bound during training:

```python
"""examples/adaptive_clipping.py"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face import DPTrainer
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    privacy_args = PrivacyArguments(
        target_epsilon=8.0,
        per_sample_max_grad_norm=1.0,
        clipping="adaptive",
        target_unclipped_quantile=0.5,
        clipbound_learning_rate=0.2,
        min_clipbound=0.05,
        max_clipbound=100.0,
    )

    training_args = TrainingArguments(
        output_dir="./output/adaptive-dp",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        report_to="none",
    )

    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # your dataset here
        privacy_args=privacy_args,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
```
