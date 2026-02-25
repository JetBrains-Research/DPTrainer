from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TrainingConfig:
    """PyTorch-centric training configuration with HuggingFace TrainingArguments duck-typing compatibility.

    Attributes:
        num_train_epochs (float): Total number of training epochs to perform. Defaults to 1.0.
        per_device_train_batch_size (int): Batch size per GPU/TPU core/CPU for training. Defaults to 16.
        gradient_accumulation_steps (int): Number of updates steps to accumulate before performing a backward/update pass. Defaults to 1.
        learning_rate (float): The initial learning rate for AdamW. Defaults to 2e-4.
        weight_decay (float): Weight decay for AdamW if we apply some. Defaults to 0.01.
        warmup_ratio (float): Linear warmup over warmup_ratio fraction of total steps. Defaults to 0.15.
        adam_beta1 (float): Beta1 for AdamW optimizer. Defaults to 0.9.
        adam_beta2 (float): Beta2 for AdamW optimizer. Defaults to 0.999.
        adam_epsilon (float): Epsilon for AdamW optimizer. Defaults to 1e-8.
        lr_scheduler_type (str): The scheduler type to use. Defaults to "linear".
        lr_scheduler_kwargs (dict): Extra arguments for the scheduler.
        eval_steps (int): Number of update steps between two evaluations. Defaults to 50.
        per_device_eval_batch_size (int): Batch size per GPU/TPU core/CPU for evaluation. Defaults to 16.
        logging_steps (int): Number of update steps between two logs. Defaults to 500.
        save_steps (int): Number of updates steps before two checkpoint saves. Defaults to 500.
        output_dir (str): The output directory where the model predictions and checkpoints will be written. Defaults to "trainer_output".
    """

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
