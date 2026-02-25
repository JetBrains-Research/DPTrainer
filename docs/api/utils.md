# Utility Helpers

## Model Conversion

### `convert_to_static`

`jbr.fed.dp_training.utils.convert_to_static`

```python
from jbr.fed.dp_training.utils import convert_to_static
```

```python
def convert_to_static(dataset: Union[IterableDataset, Dataset], num_samples: int = None) -> Dataset
```

Converts an `IterableDataset` (or any `Dataset`) to a static in-memory `Dataset`. Optionally takes only the first `num_samples` elements.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Union[IterableDataset, Dataset]` | | The dataset to convert |
| `num_samples` | `int` | `None` | If provided, only take this many samples |

**Returns:** A `datasets.Dataset` with all samples materialized in memory.

> **Source:** [`src/jbr/fed/dp_training/utils/convert_to_static.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/utils/convert_to_static.py)

---

## Loss Function Setup

### `set_loss_function_recursively`

`jbr.fed.dp_training.utils.set_loss_function_recursively`

```python
from jbr.fed.dp_training.utils import set_loss_function_recursively
```

```python
def set_loss_function_recursively(model, new_loss_function, max_depth=10) -> None
```

Recursively sets the `loss_function` property on all models in the wrapper hierarchy. Handles various model wrappers including `GradSampleModule`, `PeftModelForCausalLM`, `LoraModel`, and `DistributedDataParallel`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | | | The wrapped model to process |
| `new_loss_function` | | | The new loss function to set |
| `max_depth` | `int` | `10` | Maximum recursion depth to prevent infinite loops |

**Raises:** `ValueError` if maximum depth is reached or a circular reference is detected.

> **Source:** [`src/jbr/fed/dp_training/utils/set_loss_function.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/utils/set_loss_function.py)
