"""Privatize a third-party Trainer with differential privacy.

Demonstrates using privatize_trainer to add DP to Seq2SeqTrainer
without modifying its source code.

Usage:
    python examples/privatize_seq2seq.py
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face.utils import privatize_trainer


def main():
    model_name = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load a dummy dataset for testing
    raw_dataset = load_dataset("imdb", split="train[:16]")

    def preprocess_function(examples):
        inputs = ["classify: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        # Convert labels to strings for T5 seq2seq generation
        labels = tokenizer([str(label) for label in examples["label"]], max_length=8, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_dataset.map(preprocess_function, batched=True)

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
        remove_unused_columns=False,
    )

    # Seq2SeqTrainer now trains with differential privacy automatically
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # replace with your dataset
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
