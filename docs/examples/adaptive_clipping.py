"""Training with adaptive clipping (AdaClip).

The clip bound is adjusted dynamically during training to target
a specific fraction of unclipped samples.

Usage:
    python examples/adaptive_clipping.py
"""

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

    training_args = TrainingArguments(
        output_dir="./output/adaptive-dp",
        num_train_epochs=3,
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
    model.save_pretrained("./output/adaptive-dp/final")


if __name__ == "__main__":
    main()
