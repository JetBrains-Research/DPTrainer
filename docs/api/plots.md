# Plotting Utilities

`jbr.fed.dp_training.plots`

Privacy and loss visualization utilities. All functions save plots as PNG files and optionally CSV files to the specified output directory.

```python
from jbr.fed.dp_training.plots import plot_losses, plot_privacy_epsilon, plot_privacy_beta
```

## Functions

### `plot_losses`

```python
def plot_losses(output_dir: str, log_history) -> None
```

Plots training and evaluation loss history and saves both a PNG chart (`loss_history.png`) and a CSV file (`loss_history.csv`) to `output_dir`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_dir` | `str` | Directory to save the plot and CSV |
| `log_history` | `list[dict]` | List of log entries (from `Trainer.state.log_history`) |

### `plot_privacy_epsilon`

```python
def plot_privacy_epsilon(output_dir: str, log_history, delta) -> None
```

Plots the privacy ε history over training steps and saves a PNG chart (`eval_privacy_epsilon.png`) to `output_dir`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_dir` | `str` | Directory to save the plot |
| `log_history` | `list[dict]` | List of log entries containing `eval_privacy_epsilon` |
| `delta` | `float` | The δ value (used in the plot legend) |

### `plot_privacy_beta`

```python
def plot_privacy_beta(output_dir: str, log_history, alpha) -> None
```

Plots the privacy β history over training steps and saves a PNG chart (`eval_privacy_beta.png`) to `output_dir`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_dir` | `str` | Directory to save the plot |
| `log_history` | `list[dict]` | List of log entries containing `eval_privacy_beta` |
| `alpha` | `float` | The α value (used in the plot legend) |

---

> **Source:** [`src/jbr/fed/dp_training/plots.py`](../../src/jbr/fed/dp_training/plots.py)
