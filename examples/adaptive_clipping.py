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
        train_dataset=None,  # your dataset here
        privacy_args=privacy_args,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
