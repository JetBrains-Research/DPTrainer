"""Training with the Connect-the-Dots (CTD) accountant and beta targets.

Uses error rate targets (alpha, beta) instead of epsilon for privacy calibration.

Usage:
    python examples/ctd_accountant.py
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
