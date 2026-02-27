"""Tests for set_loss_function utility."""

import torch
import torch.nn as nn
import pytest

from dptrainer.utils.set_loss_function import set_loss_function, _unwrap_peft


# ---------------------------------------------------------------------------
# Helpers – lightweight fakes that mimic real wrapper hierarchies.
# All wrappers extend nn.Module so transformers unwrap_model can inspect them.
# ---------------------------------------------------------------------------

def _make_pretrained(loss_fn=None):
    """Create a minimal PreTrainedModel instance."""
    from transformers.modeling_utils import PreTrainedModel
    from transformers import PretrainedConfig

    class MinimalModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            super().__init__(PretrainedConfig())

    m = MinimalModel()
    m.loss_function = loss_fn
    return m


class FakePlainModel(nn.Module):
    """A non-PreTrainedModel that has a loss_function attribute."""

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_function = loss_fn


class FakeGradSampleModule(nn.Module):
    """Mimics Opacus GradSampleModule: stores inner model as _module."""

    def __init__(self, inner):
        super().__init__()
        # Store as plain attribute, NOT as nn submodule
        object.__setattr__(self, '_module', inner)


class FakeDDP(nn.parallel.DataParallel):
    """Mimics DistributedDataParallel by inheriting DataParallel.

    transformers unwrap_model checks isinstance against DataParallel.
    """

    def __init__(self, inner):
        # Bypass DataParallel.__init__ — just set .module directly
        nn.Module.__init__(self)
        self.module = inner


class FakePeft(nn.Module):
    """Mimics a PEFT wrapper with base_model.model and _peft_config."""

    def __init__(self, inner):
        super().__init__()
        # Use a simple namespace as base_model to avoid nn.Module registration
        self.base_model = _Namespace(model=inner)
        self._peft_config = True


class FakeLoraModel(nn.Module):
    """Mimics a LoraModel PEFT wrapper."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = _Namespace(model=inner)
        self._peft_config = True


class _Namespace:
    """Simple attribute holder to avoid nn.Module submodule registration."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Make class names start with "Peft" for duck-typing detection in _unwrap_peft
FakePeft.__name__ = "PeftModelForCausalLM"
FakeLoraModel.__name__ = "PeftLoraModel"


# ---------------------------------------------------------------------------
# Tests for _unwrap_peft
# ---------------------------------------------------------------------------

class TestUnwrapPeft:
    """Unit tests for the _unwrap_peft helper."""

    def test_returns_same_model_when_no_peft(self):
        model = FakePlainModel()
        assert _unwrap_peft(model) is model

    def test_unwraps_duck_typed_peft(self):
        inner = _make_pretrained()
        wrapper = FakePeft(inner)
        assert _unwrap_peft(wrapper) is inner

    def test_returns_model_without_peft_markers(self):
        """A model with base_model.model but no _peft_config is NOT unwrapped."""

        class NotPeft(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.base_model = _Namespace(model=inner)

        inner = _make_pretrained()
        wrapper = NotPeft(inner)
        assert _unwrap_peft(wrapper) is wrapper


# ---------------------------------------------------------------------------
# Tests for set_loss_function – bare model
# ---------------------------------------------------------------------------

class TestSetLossFunctionBareModel:
    """set_loss_function on a model without any wrappers."""

    def test_sets_on_pretrained_model(self):
        model = _make_pretrained(loss_fn="old")
        new_fn = lambda x: x

        set_loss_function(model, new_fn)

        assert model.loss_function is new_fn

    def test_sets_on_model_with_loss_function_attr(self):
        """A non-PreTrainedModel that has loss_function should still work."""
        model = FakePlainModel(loss_fn="old")
        set_loss_function(model, "new")
        assert model.loss_function == "new"

    def test_raises_when_no_loss_function(self):
        class NoLossModel(nn.Module):
            def __init__(self):
                super().__init__()

        model = NoLossModel()
        with pytest.raises(ValueError, match="Could not find a model with loss_function"):
            set_loss_function(model, lambda x: x)


# ---------------------------------------------------------------------------
# Tests for set_loss_function – single wrappers
# ---------------------------------------------------------------------------

class TestSetLossFunctionSingleWrapper:
    """set_loss_function with exactly one wrapper layer."""

    def test_unwraps_grad_sample_module(self):
        """GradSampleModule(_module) → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        wrapped = FakeGradSampleModule(inner)

        set_loss_function(wrapped, "new")

        assert inner.loss_function == "new"

    def test_unwraps_ddp(self):
        """DistributedDataParallel(module) → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        wrapped = FakeDDP(inner)

        set_loss_function(wrapped, "new")

        assert inner.loss_function == "new"

    def test_unwraps_peft(self):
        """PeftModel(base_model.model) → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        wrapped = FakePeft(inner)

        set_loss_function(wrapped, "new")

        assert inner.loss_function == "new"

    def test_unwraps_lora(self):
        """LoraModel(base_model.model) → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        wrapped = FakeLoraModel(inner)

        set_loss_function(wrapped, "new")

        assert inner.loss_function == "new"


# ---------------------------------------------------------------------------
# Tests for set_loss_function – nested / combined wrappers
# ---------------------------------------------------------------------------

class TestSetLossFunctionNestedWrappers:
    """set_loss_function with multiple wrapper layers stacked."""

    def test_grad_sample_then_ddp_then_peft(self):
        """GradSampleModule → DDP → PeftModel → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        peft = FakePeft(inner)
        ddp = FakeDDP(peft)
        gsm = FakeGradSampleModule(ddp)

        set_loss_function(gsm, "new")

        assert inner.loss_function == "new"

    def test_grad_sample_then_peft(self):
        """GradSampleModule → PeftModel → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        peft = FakePeft(inner)
        gsm = FakeGradSampleModule(peft)

        set_loss_function(gsm, "new")

        assert inner.loss_function == "new"

    def test_ddp_then_peft(self):
        """DDP → PeftModel → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        peft = FakePeft(inner)
        ddp = FakeDDP(peft)

        set_loss_function(ddp, "new")

        assert inner.loss_function == "new"

    def test_grad_sample_then_ddp(self):
        """GradSampleModule → DDP → PreTrainedModel (no PEFT)."""
        inner = _make_pretrained(loss_fn="old")
        ddp = FakeDDP(inner)
        gsm = FakeGradSampleModule(ddp)

        set_loss_function(gsm, "new")

        assert inner.loss_function == "new"


# ---------------------------------------------------------------------------
# Tests for set_loss_function – edge cases
# ---------------------------------------------------------------------------

class TestSetLossFunctionEdgeCases:
    """Edge cases and boundary conditions."""

    def test_loss_function_set_to_none(self):
        """Setting loss_function to None should work."""
        model = _make_pretrained(loss_fn="old")
        set_loss_function(model, None)
        assert model.loss_function is None

    def test_loss_function_set_to_string(self):
        """Setting a string identifier as loss function."""
        model = _make_pretrained(loss_fn="old")
        set_loss_function(model, "cross_entropy")
        assert model.loss_function == "cross_entropy"

    def test_idempotent_double_set(self):
        """Calling set_loss_function twice keeps the last value."""
        model = _make_pretrained(loss_fn="first")
        set_loss_function(model, "second")
        set_loss_function(model, "third")
        assert model.loss_function == "third"

    def test_non_pretrained_with_loss_function_attr(self):
        """A non-PreTrainedModel that has loss_function should still work."""
        model = FakePlainModel(loss_fn="old")
        set_loss_function(model, "new")
        assert model.loss_function == "new"

    def test_deeply_nested_all_wrappers(self):
        """GradSampleModule → DDP → LoraModel → PreTrainedModel."""
        inner = _make_pretrained(loss_fn="old")
        lora = FakeLoraModel(inner)
        ddp = FakeDDP(lora)
        gsm = FakeGradSampleModule(ddp)

        set_loss_function(gsm, "new")

        assert inner.loss_function == "new"
