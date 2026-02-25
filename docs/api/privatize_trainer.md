# privatize_trainer

`jbr.fed.dp_training.hugging_face.utils.privatize_trainer.privatize_trainer`

Convert a Hugging Face `Trainer`-based class to use `DPTrainer` as its base class. Recursively modifies the inheritance hierarchy, replacing any inheritance from `transformers.Trainer` with `DPTrainer` to enable differential privacy training capabilities.

```python
from jbr.fed.dp_training.hugging_face.utils import privatize_trainer
```

## Signature

```python
def privatize_trainer(cls, default_privacy_args=None)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `cls` | `type` | The class to be modified. Must be a subclass of `transformers.Trainer` |
| `default_privacy_args` | `PrivacyArguments` | Privacy arguments to use as a default for the class |

## Example

```python
from transformers import Seq2SeqTrainer
from jbr.fed.dp_training.hugging_face.utils import privatize_trainer

privatize_trainer(Seq2SeqTrainer)
# Seq2SeqTrainer now inherits from DPTrainer instead of Trainer
```

---

> **Source:** [`src/jbr/fed/dp_training/hugging_face/utils/privatize_trainer.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/hugging_face/utils/privatize_trainer.py)
