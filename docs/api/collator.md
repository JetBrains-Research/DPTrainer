# DataCollatorForCausalLM

`jbr.fed.dp_training.hugging_face.patched.collator.DataCollatorForCausalLM`

Data collator that dynamically pads inputs and adds labels shifted by one element to the left. Extends `DataCollatorMixin`.

```python
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `tokenizer` | `PreTrainedTokenizerBase` | | The tokenizer used for encoding the data |
| `tokenize` | `Union[bool, str]` | `False` | Whether to tokenize inputs |
| `padding` | `Union[bool, str, PaddingStrategy]` | `True` | Padding strategy: `True`/`"longest"`, `"max_length"`, or `False`/`"do_not_pad"` |
| `max_length` | `Optional[int]` | `None` | Maximum length of the returned list and optionally padding length |
| `pad_to_multiple_of` | `Optional[int]` | `None` | If set, pads the sequence to a multiple of the provided value |
| `return_tensors` | `str` | `"pt"` | The type of Tensor to return (`"np"`, `"pt"`, or `"tf"`) |

## Behavior

When called with a list of feature dictionaries, the collator:

1. Filters out features where `attention_mask` is all zeros
2. Pads all features to the same length using the tokenizer
3. Creates `labels` from `input_ids` if not already present (masking padding tokens with `-100`)
4. Adds `position_ids` if not present

Returns a dictionary with keys: `input_ids`, `attention_mask`, `labels`, `position_ids`.

---

> **Source:** [`src/jbr/fed/dp_training/hugging_face/patched/collator.py`](https://github.com/JetBrains-Research/DPTrainer/blob/main/src/jbr/fed/dp_training/hugging_face/patched/collator.py)
