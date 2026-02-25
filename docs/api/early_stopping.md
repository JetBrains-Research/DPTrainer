# EarlyStoppingCallback

`jbr.fed.dp_training.hugging_face.patched.early_stopping.EarlyStoppingCallback`

Checkpoint-aware early stopping callback. Extends `transformers.EarlyStoppingCallback` to correctly handle early stopping when restoring from a checkpoint — if the patience counter has already been exceeded at the start of training, it immediately stops.

```python
from jbr.fed.dp_training.hugging_face.patched import EarlyStoppingCallback
```

## Behavior

Overrides `on_train_begin` to check whether `early_stopping_patience_counter >= early_stopping_patience` at the start of training. This ensures that when resuming from a checkpoint where patience was already exhausted, training stops immediately rather than continuing.

All other behavior is inherited from `transformers.EarlyStoppingCallback`.

---

> **Source:** [`src/jbr/fed/dp_training/hugging_face/patched/early_stopping.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/hugging_face/patched/early_stopping.py)
