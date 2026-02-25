# PyTorch Training Loop

`jbr.fed.dp_training.pytorch`

Low-level PyTorch DP training loop with built-in support for multi-GPU training, privacy accounting, checkpointing, early stopping, and TensorBoard logging.

```python
from jbr.fed.dp_training.pytorch import custom_training_loop
```

## `custom_training_loop`

```python
def custom_training_loop(
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Union[Dataset, IterableDataset],
    eval_dataset: Union[Dataset, IterableDataset],
    training_arguments: TrainingArgumentsProtocol,
    privacy_arguments: PrivacyArguments = None,
    model_input_names: List[str] = None,
    early_stopping_patience: Optional[int] = None,
) -> tuple[list[dict], Accountant]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `Module` | | PyTorch model to train |
| `tokenizer` | `PreTrainedTokenizerBase` | | Tokenizer for data collation |
| `train_dataset` | `Union[Dataset, IterableDataset]` | | Training dataset |
| `eval_dataset` | `Union[Dataset, IterableDataset]` | | Evaluation dataset |
| `training_arguments` | `TrainingArgumentsProtocol` | | Training configuration (compatible with HF `TrainingArguments` or `TrainingConfig`) |
| `privacy_arguments` | `PrivacyArguments` | `None` | Privacy configuration. Defaults to low-privacy if not provided |
| `model_input_names` | `List[str]` | `None` | Model input keys to keep from the dataset. Defaults to `["input_ids", "attention_mask", "labels"]` |
| `early_stopping_patience` | `Optional[int]` | `None` | Number of evaluations without improvement before stopping |

**Returns:** A tuple of `(log_history, accountant)` where `log_history` is a list of training/evaluation log dictionaries and `accountant` is the privacy accountant with the final state.

### Features

- **Multi-GPU support** — automatically uses `torch.multiprocessing.spawn` with `DPDDP` when multiple GPUs are available
- **Checkpoint resume** — saves and restores model, optimizer, scheduler, privacy engine, and training state
- **Privacy budget enforcement** — stops training when ε or β budget is exhausted
- **Early stopping** — stops training when evaluation loss stops improving
- **TensorBoard logging** — logs loss, learning rate, gradient norms, and privacy metrics

## `TrainingConfig`

`jbr.fed.dp_training.pytorch.training_args.TrainingConfig`

PyTorch-centric training configuration dataclass with HuggingFace `TrainingArguments` duck-typing compatibility.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_train_epochs` | `float` | `1.0` | Number of training epochs |
| `per_device_train_batch_size` | `int` | `16` | Training batch size per device |
| `gradient_accumulation_steps` | `int` | `1` | Gradient accumulation steps |
| `learning_rate` | `float` | `0.0002` | Learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `warmup_ratio` | `float` | `0.15` | Warmup ratio |
| `adam_beta1` | `float` | `0.9` | Adam β1 |
| `adam_beta2` | `float` | `0.999` | Adam β2 |
| `adam_epsilon` | `float` | `1e-8` | Adam ε |
| `lr_scheduler_type` | `str` | `"linear"` | Learning rate scheduler type |
| `lr_scheduler_kwargs` | `dict` | `{}` | Additional scheduler keyword arguments |
| `eval_steps` | `int` | `50` | Evaluate every N steps |
| `per_device_eval_batch_size` | `int` | `16` | Evaluation batch size per device |
| `logging_steps` | `int` | `500` | Log every N steps |
| `save_steps` | `int` | `500` | Save checkpoint every N steps |
| `output_dir` | `str` | `"trainer_output"` | Output directory |

---

> **Source:** [`src/jbr/fed/dp_training/pytorch/loop.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/pytorch/loop.py), [`src/jbr/fed/dp_training/pytorch/training_args.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/pytorch/training_args.py)
