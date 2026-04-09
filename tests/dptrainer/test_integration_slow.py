"""Integration tests with real HuggingFace models.

These tests are slower and require actual model training.
Run with: RUN_SLOW_TESTS=1 pytest tests/test_integration.py -v
"""

import os
import pytest
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, PeftModel

from dptrainer import PrivacyArguments
from dptrainer.trainer import DPTrainer


# Skip slow tests by default
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow tests are skipped by default. Set RUN_SLOW_TESTS=1 to run them.",
)

@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    """Fixture providing GPT-2 small model and tokenizer."""
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["c_proj"]))

    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


@pytest.fixture(scope="module")
def code_dataset(gpt2_model_and_tokenizer):
    """Fixture providing a small code dataset (Python)."""
    _, tokenizer = gpt2_model_and_tokenizer

    # Load a small subset of a code dataset
    dataset = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)

    # Take only 100 examples for speed
    dataset = dataset.take(50)
    dataset = dataset.map(lambda x: {"text": x["content"]})

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    static_dataset = Dataset.from_list(list(tokenized_dataset))

    return static_dataset.train_test_split(test_size=0.2, seed=42)


@pytest.fixture
def data_collator(gpt2_model_and_tokenizer):
    """Fixture providing data collator for language modeling."""
    _, tokenizer = gpt2_model_and_tokenizer
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal LM, not masked LM
    )


class TestDPTrainerIntegration:
    """Integration tests with real GPT-2 model and code dataset."""

    def test_basic_training_with_gpt2(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that DPTrainer can train GPT-2 on code data."""
        model, _ = gpt2_model_and_tokenizer

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=3,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=2,
            eval_steps=2,
            eval_strategy="steps",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            noise_multiplier=0.1,
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode="hooks",
        )

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            eval_dataset=code_dataset["test"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Verify training completed
        assert train_result is not None
        assert trainer.state.global_step == 10

        # Verify privacy metrics are logged
        logs = trainer.state.log_history
        privacy_metrics = [log for log in logs if "privacy_epsilon" in log or "eval_privacy_epsilon" in log]
        assert len(privacy_metrics) > 0

        # Verify epsilon is positive
        final_epsilon = privacy_metrics[-1].get("privacy_epsilon") or privacy_metrics[-1].get("eval_privacy_epsilon")
        assert final_epsilon > 0

    def test_checkpoint_save_and_resume(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that checkpointing works and training can be resumed."""
        model, _ = gpt2_model_and_tokenizer

        checkpoint_dir = tmp_path / "checkpoints"

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=2,
            save_strategy="steps",
            save_steps=5,
            save_total_limit=2,
            report_to="none",
            disable_tqdm=True,
            restore_callback_states_from_checkpoint=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            noise_multiplier=0.1,
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
        )

        # First training session - train for 5 steps
        training_args.max_steps = 5
        trainer1 = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        trainer1.train()

        # Verify checkpoint was saved
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0

        checkpoint_path = str(checkpoints[0])

        # Get privacy epsilon at checkpoint (if available)
        epsilon_at_checkpoint = 0
        for log in trainer1.state.log_history:
            if "privacy_epsilon" in log or "eval_privacy_epsilon" in log:
                epsilon_at_checkpoint = log.get("privacy_epsilon") or log.get("eval_privacy_epsilon")

        # Resume training from checkpoint
        model2, _ = gpt2_model_and_tokenizer
        training_args.max_steps = 10

        trainer2 = DPTrainer(
            model=model2,
            args=training_args,
            train_dataset=code_dataset["train"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        train_result = trainer2.train(resume_from_checkpoint=checkpoint_path)

        # Verify training resumed and completed
        assert train_result is not None
        assert trainer2.state.global_step == 10

    def test_model_save_and_load(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that the trained model can be saved and loaded correctly."""
        model, tokenizer = gpt2_model_and_tokenizer

        model_dir = tmp_path / "trained_model"

        training_args = TrainingArguments(
            output_dir=str(tmp_path / "output"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=5,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            noise_multiplier=0.1,
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
        )

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Detach model from DP wrapper
        detached_model = trainer.detach_model()

        # Save the model
        detached_model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        # Verify files were saved
        assert (model_dir / "adapter_config.json").exists()
        assert (model_dir / "adapter_model.safetensors").exists() or (model_dir / "adapter_model.bin").exists()

        # Load the base model and adapter
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        loaded_model = PeftModel.from_pretrained(base_model, str(model_dir))
        loaded_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Verify loaded model works
        test_input = "def hello_world():"
        inputs = loaded_tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            outputs = loaded_model(**inputs)

        assert outputs is not None
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 1  # batch size

    def test_model_inference_after_training(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that the model can generate code after DP training."""
        model, tokenizer = gpt2_model_and_tokenizer

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=2,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            noise_multiplier=0.1,
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
        )

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Detach model
        detached_model = trainer.detach_model()
        detached_model.eval()

        # Test generation
        prompt = "def calculate_sum("
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move inputs to the same device as the model
        device = next(detached_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = detached_model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Verify generation worked
        assert generated_text is not None
        assert len(generated_text) > len(prompt)
        assert generated_text.startswith(prompt)

    def test_privacy_budget_tracking(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that privacy budget is tracked correctly throughout training."""
        model, _ = gpt2_model_and_tokenizer

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=1,
            eval_steps=1,
            eval_strategy="steps",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            noise_multiplier=0.8,
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
        )

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            eval_dataset=code_dataset["test"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Get all privacy metrics
        logs = trainer.state.log_history
        privacy_epsilons = [
            log.get("privacy_epsilon") or log.get("eval_privacy_epsilon")
            for log in logs
            if "privacy_epsilon" in log or "eval_privacy_epsilon" in log
        ]

        # Verify epsilon is monotonically increasing
        assert len(privacy_epsilons) > 0
        for i in range(1, len(privacy_epsilons)):
            assert privacy_epsilons[i] >= privacy_epsilons[i-1], \
                f"Epsilon should be monotonically increasing: {privacy_epsilons}"

    def test_target_epsilon_calculation(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test that noise multiplier is calculated from target epsilon."""
        model, _ = gpt2_model_and_tokenizer

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=2,
            eval_steps=2,
            eval_strategy="steps",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments(
            accountant="rdp",
            target_epsilon=10.0,  # Set target epsilon instead of noise multiplier
            per_sample_max_grad_norm=1.0,
            poisson_sampling=False,
        )

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            eval_dataset=code_dataset["test"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Verify noise multiplier was calculated
        assert trainer.privacy_args.noise_multiplier is not None
        assert trainer.privacy_args.noise_multiplier > 0

        # Train
        trainer.train()

        # Verify final epsilon is tracked
        logs = trainer.state.log_history
        privacy_metrics = [log for log in logs if "privacy_epsilon" in log or "eval_privacy_epsilon" in log]
        assert len(privacy_metrics) > 0

    def test_low_privacy_mode(
        self, gpt2_model_and_tokenizer, code_dataset, data_collator, tmp_path
    ):
        """Test training with low privacy (no noise) for debugging."""
        model, _ = gpt2_model_and_tokenizer

        training_args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_steps=5,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        )

        privacy_args = PrivacyArguments.low_privacy()

        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=code_dataset["train"],
            privacy_args=privacy_args,
            data_collator=data_collator,
        )

        # Verify no noise
        assert trainer.privacy_args.noise_multiplier == 0.0

        # Train
        train_result = trainer.train()

        # Verify training completed
        assert train_result is not None
        assert trainer.state.global_step == 5
