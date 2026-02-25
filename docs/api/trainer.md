# DPTrainer

`jbr.fed.dp_training.hugging_face.trainer.DPTrainer`

Hugging Face `Trainer` with Differential Privacy support. Extends `transformers.Trainer`.

```python
from jbr.fed.dp_training.hugging_face import DPTrainer
```

## Constructor

```python
DPTrainer(
    model: Union[PreTrainedModel, nn.Module] = None,
    args: TrainingArguments = None,
    train_dataset: Union[datasets.Dataset, torch.utils.data.Dataset] = None,
    privacy_args: PrivacyArguments = None,
    compute_metrics: Optional[Callable] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | `Union[PreTrainedModel, nn.Module]` | Model to train |
| `args` | `TrainingArguments` | Training arguments |
| `train_dataset` | `Union[datasets.Dataset, torch.utils.data.Dataset]` | Training dataset |
| `privacy_args` | `PrivacyArguments` | Privacy arguments for differential private training |
| `compute_metrics` | `Optional[Callable]` | Custom evaluation metrics |
| `callbacks` | `Optional[list[TrainerCallback]]` | Training callbacks |
| `**kwargs` | | Additional keyword arguments passed to `Trainer` |

The constructor automatically:

- Validates privacy arguments and dataset compatibility
- Calculates the noise multiplier if not explicitly set
- Wraps the model in an Opacus `GradSampleModule`
- Creates a `DPCallback` for privacy accounting
- Wraps the optimizer in a `DPOptimizer`

## Methods

### `create_optimizer`

```python
def create_optimizer(self) -> DPOptimizer
```

Creates and returns a DP-wrapped optimizer. Selects the appropriate `DPOptimizer` subclass based on the clipping strategy (`flat`, `adaptive`, or `per_layer`).

### `get_train_dataloader`

```python
def get_train_dataloader(self) -> torch.utils.data.DataLoader
```

Returns a DP-compatible training dataloader. Applies Poisson sampling if enabled and wraps the dataloader with Opacus `BatchMemoryManager`.

### `detach_model`

```python
def detach_model(self) -> nn.Module
```

Detach the model from the controller and return the model. Cleans up resources or connections associated with the private trainer and detaches the managed model for further usage.

---

> **Source:** [`src/jbr/fed/dp_training/hugging_face/trainer.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/hugging_face/trainer.py)
