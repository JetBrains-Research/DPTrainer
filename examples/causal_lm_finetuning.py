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
