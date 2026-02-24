from functools import wraps

from opacus.grad_sample import AbstractGradSampleModule
from transformers import Trainer

from jbr.fed.dp_training.hugging_face import DPTrainer


def privatize_trainer(cls, default_privacy_args = None):
    """
    Convert a Hugging Face Trainer-based class to use DPTrainer as its base class.

    This function recursively modifies the inheritance hierarchy of the given class
    and all its subclasses, replacing any inheritance from `transformers.Trainer`
    with `DPTrainer` to enable differential privacy training capabilities.

    Args:
        cls: The class to be modified. Must be a subclass of transformers.Trainer.
        default_privacy_args: Privacy arguments to be passed to cls to use as a default.
    Example:
        >>> from transformers import Seq2SeqTrainer
        >>> from jbr.fed.dp_training.hugging_face import privatize_trainer
        >>>
        >>> privatize_trainer(Seq2SeqTrainer)
        # Seq2SeqTrainer now inherits from DPTrainer instead of Trainer
    """
    _change_base_recursively(cls, Trainer, DPTrainer)

    setattr(cls, "default_privacy_args", default_privacy_args)


def _change_base_recursively(cls, old_base, new_base, visited=None):
    """
    Recursively replace old_base with new_base in cls and all its subclasses.
    """
    if cls == old_base:
        raise ValueError(f"Cannot replace {old_base} with {new_base} inplace or in the top of the hierarchy. ")
    if visited is None:
        visited = { new_base }

    if cls in visited:
        return
    visited.add(cls)

    # Change bases for the current class
    if old_base in cls.__bases__:
        new_bases = tuple(new_base if base == old_base else base for base in cls.__bases__)
        cls.__bases__ = new_bases

    # Recursively process all subclasses
    for baseclass in cls.__bases__:
        _change_base_recursively(baseclass, old_base, new_base, visited)