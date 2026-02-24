from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TrainingConfig:
    """PyTorch-centric training configuration with HuggingFace TrainingArguments duck-typing compatibility."""

    # Training hyperparameters (HuggingFace compatible names)
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    warmup_ratio: float = 0.15

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate scheduler
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict = field(default_factory=dict)

    # Evaluation
    eval_steps: int = 50
    per_device_eval_batch_size: int = 16

    # Logging and checkpointing
    logging_steps: int = 500
    save_steps: int = 500

    # Output directory
    output_dir: str = "trainer_output"


class TrainingArgumentsProtocol(Protocol):
    """Protocol for duck-typing compatibility with HuggingFace TrainingArguments."""

    # Required attributes for compatibility
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float

    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float

    lr_scheduler_type: str
    lr_scheduler_kwargs: dict

    eval_steps: int
    per_device_eval_batch_size: int

    logging_steps: int
    save_steps: int

    output_dir: str
