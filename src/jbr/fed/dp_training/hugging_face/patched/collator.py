import logging
from dataclasses import dataclass
from typing import Union, Optional, Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, DataCollatorMixin
from transformers.utils import PaddingStrategy

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForCausalLM(DataCollatorMixin):
    """Data collator that dynamically pads the inputs received and adds labels shifted by one element to the left.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        tokenize (`bool` or `str`, *optional*, defaults to `False`):
            Whether to tokenize the inputs before collating.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    Returns:
        A dictionary of a padded input_ids tensor, attention_mask tensor, and labels tensor shifted by one
        element to inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    tokenize: Union[bool, str] = False
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # noinspection PyMethodOverriding
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = list(filter(lambda x: sum(x["attention_mask"]) != 0, features))
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" in features[0]:
            labels = batch["labels"]
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100

        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "position_ids": batch["position_ids"],
        }
