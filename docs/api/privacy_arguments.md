# PrivacyArguments

`jbr.fed.dp_training.privacy_arguments.PrivacyArguments`

Dataclass for all privacy-related training parameters.

```python
from jbr.fed.dp_training import PrivacyArguments
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `accountant` | `str` | `"rdp"` | Accountant mechanism to use for DP training |
| `grad_sample_mode` | `str` | `"hooks"` | Grad sample mode of Opacus (`"hooks"` or `"ghost"`) |
| `per_sample_max_grad_norm` | `float` | `0.5` | Max per sample clip norm |
| `clipping` | `str` | `"flat"` | Clipping strategy (`"flat"`, `"adaptive"`, or `"per_layer"`) |
| `poisson_sampling` | `bool` | `True` | Use Poisson sampling, use standard batches otherwise |
| `min_clipbound` | `float` | `0.05` | Min clip bound of the AdaClip algorithm |
| `max_clipbound` | `float` | `1e8` | Max clip bound of the AdaClip algorithm |
| `clipbound_learning_rate` | `float` | `0.2` | Learning rate of the AdaClip algorithm |
| `target_unclipped_quantile` | `float` | `0.5` | Target fraction of unclipped samples per batch (AdaClip) |
| `unclipped_num_std` | `float` | `1.0` | Standard deviation of the unclipped number noise (AdaClip) |
| `noise_multiplier` | `Optional[float]` | `None` | Noise multiplier for DP training |
| `target_epsilon` | `Optional[float]` | `None` | Target epsilon at end of training (mutually exclusive with noise multiplier) |
| `target_delta` | `Optional[float]` | `None` | Target delta, defaults to `1/N` |
| `target_alpha` | `Optional[float]` | `0.0001` | Target FPR for beta search |
| `target_beta` | `Optional[float]` | `None` | Target FNR |

## Methods

### `low_privacy`

```python
@classmethod
def low_privacy(cls) -> PrivacyArguments
```

Returns a `PrivacyArguments` instance with no privacy (`noise_multiplier=0.0`, `poisson_sampling=False`, `accountant="rdp"`).

### `precalculate`

```python
def precalculate(self, num_samples: int, sample_rate: float, steps: int) -> None
```

Auto-calculates the noise multiplier based on the privacy target. Sets `target_delta` to `1/num_samples` if not provided. The priority is:

1. **Explicit `noise_multiplier`** — used directly if already set.
2. **`target_epsilon`** — noise is calibrated to achieve (ε, δ)-DP using the specified accountant.
3. **`target_beta`** — noise is calibrated to achieve (α, β) error rates using [riskcal](https://github.com/microsoft/riskcal).
4. **None specified** — defaults to `noise_multiplier=0.0`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `num_samples` | `int` | Size of the training dataset |
| `sample_rate` | `float` | Batch sampling rate |
| `steps` | `int` | Total number of training steps |

---

> **Source:** [`src/jbr/fed/dp_training/privacy_arguments.py`](../../src/jbr/fed/dp_training/privacy_arguments.py)
