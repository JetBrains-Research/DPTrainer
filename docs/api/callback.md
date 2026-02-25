# DPCallback

`jbr.fed.dp_training.hugging_face.callback.DPCallback`

Trainer callback that makes `transformers.Trainer` compatible with Opacus. Handles privacy accounting, budget enforcement, and optimizer hooks. Extends `TrainerCallback` and `ExportableState`.

```python
from jbr.fed.dp_training.hugging_face.callback import DPCallback
```

## Constructor

```python
DPCallback(
    accountant: str,
    gradient_accumulation_steps: int,
    target_delta: float,
    target_alpha: float,
    max_epsilon: float = None,
    min_beta: float = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `accountant` | `str` | | Privacy accountant mechanism (`"rdp"` or `"ctd"`) |
| `gradient_accumulation_steps` | `int` | | Number of gradient accumulation steps |
| `target_delta` | `float` | | Target δ for privacy accounting |
| `target_alpha` | `float` | | Target FPR for β search |
| `max_epsilon` | `float` | `None` | Maximum ε — training stops when exceeded |
| `min_beta` | `float` | `None` | Minimum β — training stops when exceeded |

## Methods

### `get_optimizer_callback`

```python
def get_optimizer_callback(self, sample_rate: float)
```

Returns the optimizer hook function for the privacy accountant.

### `on_train_begin`

```python
def on_train_begin(self, args, state, control, **kwargs)
```

Checks if the privacy budget is already exceeded at the start of training.

### `on_step_begin`

```python
def on_step_begin(self, args, state, control, optimizer=None, **kwargs)
```

Cleans up extra elements in the optimizer's step skip queue.

### `on_substep_end`

```python
def on_substep_end(self, args, state, control, optimizer=None, **kwargs)
```

Steps the optimizer and clears gradients after each gradient accumulation substep.

### `on_step_end`

```python
def on_step_end(self, args, state, control, optimizer=None, **kwargs)
```

Clears gradients after each full training step.

### `on_evaluate`

```python
def on_evaluate(self, args, state, control, optimizer=None, metrics=None, **kwargs)
```

Checks if the privacy budget has been exceeded after evaluation.

### `get_privacy_metrics`

```python
def get_privacy_metrics(self) -> dict
```

Returns current privacy metrics. For the CTD accountant, returns `privacy_epsilon`, `privacy_beta`, and `privacy_advantage`. For the RDP accountant, returns `privacy_epsilon`.

### `state`

```python
def state(self) -> dict
```

Returns the serializable state of the callback (for checkpoint saving/restoring).

---

> **Source:** [`src/jbr/fed/dp_training/hugging_face/callback.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/hugging_face/callback.py)
