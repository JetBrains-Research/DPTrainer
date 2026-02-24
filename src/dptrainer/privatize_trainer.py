import warnings

from transformers import Trainer

from dptrainer.trainer import DPTrainer

_GHOST_CLIPPING_OVERRIDE_METHODS = ("compute_loss", "training_step")


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
        >>> from dptrainer import privatize_trainer
        >>>
        >>> privatize_trainer(Seq2SeqTrainer)
        # Seq2SeqTrainer now inherits from DPTrainer instead of Trainer
    """
    _change_base_recursively(cls, Trainer, DPTrainer)

    ghost_clipping_enabled = (
        default_privacy_args is not None
        and getattr(default_privacy_args, "grad_sample_mode", None) == "ghost"
    )
    if ghost_clipping_enabled:
        _warn_ghost_clipping_overrides(cls)

    setattr(cls, "default_privacy_args", default_privacy_args)


def _warn_ghost_clipping_overrides(cls):
    """
    Emit warnings if *cls* or any class between it and DPTrainer in the MRO
    overrides methods that could bypass ghost-clipping loss wrapping.
    """
    for klass in cls.__mro__:
        if klass in (DPTrainer, Trainer, object):
            break
        for method_name in _GHOST_CLIPPING_OVERRIDE_METHODS:
            if method_name in klass.__dict__:
                warnings.warn(
                    f"{klass.__qualname__} overrides '{method_name}'. "
                    f"This may bypass DPTrainer's ghost-clipping loss wrapping "
                    f"and lead to incorrect privacy gradients. "
                    f"Consider removing the override or delegating to "
                    f"super().{method_name}() to preserve differential privacy guarantees.",
                    UserWarning,
                    stacklevel=2,
                )


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
