# Configuration

## `PrivacyArguments`

All privacy-related parameters are configured through the `PrivacyArguments` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `accountant` | `"rdp"` | Privacy accountant mechanism (`"rdp"`) |
| `noise_multiplier` | `None` | Noise multiplier σ for DP-SGD. Auto-calculated if `None` |
| `target_epsilon` | `None` | Target ε at end of training (used to auto-calculate σ) |
| `target_delta` | `None` | Target δ (defaults to `1/N` where N is the dataset size) |
| `per_sample_max_grad_norm` | `0.5` | Maximum gradient norm for per-sample clipping |
| `clipping` | `"flat"` | Clipping strategy: `"flat"`, `"adaptive"`, or `"per_layer"` |
| `poisson_sampling` | `True` | Use Poisson sub-sampling for privacy amplification |
| `grad_sample_mode` | `"hooks"` | Opacus grad sample mode: `"hooks"` or `"ghost"` |

### Adaptive Clipping (AdaClip) Parameters

When using `clipping="adaptive"`, the following additional parameters are available:

| Parameter | Default | Description |
|---|---|---|
| `min_clipbound` | `0.05` | Minimum clip bound |
| `max_clipbound` | `1e8` | Maximum clip bound |
| `clipbound_learning_rate` | `0.2` | Learning rate for clip bound adaptation |
| `target_unclipped_quantile` | `0.5` | Target fraction of unclipped samples |
| `unclipped_num_std` | `1.0` | Standard deviation of unclipped number noise |

## Noise Calibration

The noise multiplier is determined automatically based on the privacy target you provide. The priority is:

1. **Explicit `noise_multiplier`** — used directly if provided.
2. **`target_epsilon`** — noise is calibrated to achieve the target (ε, δ)-DP using the specified accountant.
3. **None specified** — defaults to `noise_multiplier=0.0` (no noise, no privacy).

### Example: Epsilon-Based Calibration

```python
from dptrainer import PrivacyArguments

# Noise multiplier is auto-calculated to achieve ε=8.0
privacy_args = PrivacyArguments(
    target_epsilon=8.0,
    target_delta=1e-5,
    per_sample_max_grad_norm=1.0,
    accountant="rdp",
)
```

### Example: Explicit Noise Multiplier

```python
from dptrainer import PrivacyArguments

# Set the noise multiplier directly — no auto-calibration
privacy_args = PrivacyArguments(
    noise_multiplier=1.1,
    per_sample_max_grad_norm=1.0,
)
```

## Privacy Budget Early Stopping

`DPTrainer` automatically monitors the privacy budget during training. If a `target_epsilon` is set, training will stop early when the budget is exhausted. This is handled by the built-in `DPCallback`.

Privacy metrics (`privacy_epsilon`) are logged alongside standard training metrics during evaluation.

## Clipping Strategies

### Flat Clipping (default)

Standard per-sample gradient clipping. All gradients are clipped to the same maximum norm.

```python
privacy_args = PrivacyArguments(
    clipping="flat",
    per_sample_max_grad_norm=1.0,
)
```

### Adaptive Clipping (AdaClip)

The clip bound is adjusted dynamically during training to target a specific fraction of unclipped samples.

```python
privacy_args = PrivacyArguments(
    clipping="adaptive",
    per_sample_max_grad_norm=1.0,  # initial clip bound
    target_unclipped_quantile=0.5,
    clipbound_learning_rate=0.2,
)
```

### Per-Layer Clipping

Gradients are clipped independently for each layer of the model.

```python
privacy_args = PrivacyArguments(
    clipping="per_layer",
    per_sample_max_grad_norm=1.0,
)
```

## Ghost Clipping

Ghost clipping computes per-sample gradient norms without materializing full per-sample gradients, significantly reducing memory usage.

```python
privacy_args = PrivacyArguments(
    grad_sample_mode="ghost",
    per_sample_max_grad_norm=1.0,
    target_epsilon=8.0,
)
```

!!! note
    Ghost clipping requires the model to have a `loss_function` attribute. Adaptive clipping is not supported with ghost clipping and will fall back to flat clipping.

### Ghost Clipping Safety Guards

When ghost clipping is enabled, `DPTrainer` wraps the model's `loss_function` with `DPLossFastGradientClipping` so that per-sample gradient norms are computed correctly. Two safety mechanisms protect against accidental bypasses:

- **Static warning (`privatize_trainer`)** — when `privatize_trainer` is called with `grad_sample_mode="ghost"`, it inspects the trainer subclass hierarchy and emits a `UserWarning` if `compute_loss` or `training_step` is overridden. Such overrides may bypass the wrapped loss function and break privacy gradient computation.

- **Runtime validation (`DPCallback`)** — on the first training step, `DPCallback` checks that the model's `loss_function` is still an instance of `DPLossFastGradientClipping` when a ghost clipping optimizer is active. If the loss function was replaced or unwrapped after initialization, a `RuntimeError` is raised to prevent silent privacy violations.

## Distributed Training

`DPTrainer` is designed for single-GPU training. Distributed training (multi-GPU or multi-node) is **not supported** and will raise a `ValueError` during initialization if `world_size > 1`.
