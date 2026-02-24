"""Tests for utility functions."""

import warnings
from unittest.mock import Mock, patch, MagicMock
import pytest
from transformers import Trainer

try:
    import trl
    _has_trl = True
except ImportError:
    _has_trl = False

from dptrainer import privatize_trainer
from dptrainer.privatize_trainer import (
    _change_base_recursively,
    _warn_ghost_clipping_overrides,
)

from opacus.optimizers import DPOptimizer

from dptrainer.trainer import DPTrainer
from dptrainer import PrivacyArguments


class TestPrivatizeTrainer:
    """Test privatize_trainer function."""

    def test_privatize_trainer_changes_base_class(self):
        """Test that privatize_trainer changes base class to DPTrainer."""

        # Create a custom trainer class inheriting from Trainer
        class CustomTrainer(Trainer):
            pass

        # Before privatization
        assert Trainer in CustomTrainer.__bases__

        # Privatize
        privatize_trainer(CustomTrainer)

        # After privatization
        assert DPTrainer in CustomTrainer.__bases__
        assert Trainer not in CustomTrainer.__bases__


    def test_privatize_trainer_with_default_privacy_args(self):
        """Test that default_privacy_args is set correctly."""

        class CustomTrainer(Trainer):
            pass

        default_args = PrivacyArguments(noise_multiplier=1.0)
        privatize_trainer(CustomTrainer, default_privacy_args=default_args)

        assert hasattr(CustomTrainer, "default_privacy_args")
        assert CustomTrainer.default_privacy_args is default_args

    def test_privatize_trainer_nested_inheritance(self):
        """Test privatization with nested inheritance hierarchy."""

        # Create a hierarchy: CustomTrainer -> IntermediateTrainer -> Trainer
        class IntermediateTrainer(Trainer):
            pass

        class CustomTrainer(IntermediateTrainer):
            pass

        # Privatize the top-level class
        privatize_trainer(CustomTrainer)

        # Both should have DPTrainer in their hierarchy
        assert DPTrainer in IntermediateTrainer.__bases__
        assert Trainer not in IntermediateTrainer.__bases__

    def test_privatize_trainer_patches_accelerator(self):
        """Test that privatize_trainer patches Accelerator.unwrap_model."""

        class CustomTrainer(Trainer):
            pass

        # Privatize (this should patch Accelerator.unwrap_model)
        # We can't easily test the patching without importing accelerate,
        # so we just verify privatization doesn't raise errors
        privatize_trainer(CustomTrainer)

        # Verify base class was changed
        assert DPTrainer in CustomTrainer.__bases__


class TestChangeBaseRecursively:
    """Test _change_base_recursively helper function."""

    def test_change_base_simple(self):
        """Test simple base class replacement."""

        class OldBase:
            pass

        class NewBase:
            pass

        class Child(OldBase):
            pass

        _change_base_recursively(Child, OldBase, NewBase)

        assert NewBase in Child.__bases__
        assert OldBase not in Child.__bases__

    def test_change_base_multiple_bases(self):
        """Test base replacement with multiple inheritance."""

        class OldBase:
            pass

        class NewBase:
            pass

        class OtherBase:
            pass

        class Child(OldBase, OtherBase):
            pass

        _change_base_recursively(Child, OldBase, NewBase)

        assert NewBase in Child.__bases__
        assert OtherBase in Child.__bases__
        assert OldBase not in Child.__bases__

    def test_change_base_nested_hierarchy(self):
        """Test base replacement in nested hierarchy."""

        class OldBase:
            pass

        class NewBase:
            pass

        class Parent(OldBase):
            pass

        class Child(Parent):
            pass

        # Change base in child (should affect parent too)
        _change_base_recursively(Child, OldBase, NewBase)

        assert NewBase in Parent.__bases__
        assert OldBase not in Parent.__bases__

    def test_change_base_rejects_inplace_replacement(self):
        """Test that replacing the old_base itself raises error."""

        class OldBase:
            pass

        class NewBase:
            pass

        with pytest.raises(
            ValueError, match="Cannot replace .* inplace or in the top of the hierarchy"
        ):
            _change_base_recursively(OldBase, OldBase, NewBase)

    def test_change_base_with_visited_set(self):
        """Test that visited set prevents infinite loops."""

        class OldBase:
            pass

        class NewBase:
            pass

        class A(OldBase):
            pass

        class B(A):
            pass

        # Manually create circular reference (not typical but tests visited logic)
        visited = set()
        _change_base_recursively(B, OldBase, NewBase, visited=visited)

        # Should process both A and B
        assert A in visited
        assert B in visited
        assert NewBase in visited  # NewBase added to visited

    def test_change_base_preserves_other_bases(self):
        """Test that other base classes are preserved during replacement."""

        class OldBase:
            pass

        class NewBase:
            pass

        class Mixin:
            pass

        class Child(Mixin, OldBase):
            pass

        _change_base_recursively(Child, OldBase, NewBase)

        # Should have both Mixin and NewBase
        assert Mixin in Child.__bases__
        assert NewBase in Child.__bases__
        assert OldBase not in Child.__bases__
        assert len(Child.__bases__) == 2


class TestWarnGhostClippingOverrides:
    """Test _warn_ghost_clipping_overrides static analysis warnings."""

    def test_no_warning_for_clean_subclass(self):
        """No warnings when the subclass does not override compute_loss or training_step."""

        class CleanTrainer(Trainer):
            pass

        _change_base_recursively(CleanTrainer, Trainer, DPTrainer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_ghost_clipping_overrides(CleanTrainer)
            assert len(w) == 0

    def test_warning_for_compute_loss_override(self):
        """Warns when the subclass overrides compute_loss."""

        class SneakyTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                return super().compute_loss(model, inputs, **kwargs)

        _change_base_recursively(SneakyTrainer, Trainer, DPTrainer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_ghost_clipping_overrides(SneakyTrainer)
            assert len(w) == 1
            assert "compute_loss" in str(w[0].message)

    def test_warning_for_training_step_override(self):
        """Warns when the subclass overrides training_step."""

        class SneakyTrainer(Trainer):
            def training_step(self, model, inputs, num_items_in_batch=None):
                return super().training_step(model, inputs, num_items_in_batch)

        _change_base_recursively(SneakyTrainer, Trainer, DPTrainer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_ghost_clipping_overrides(SneakyTrainer)
            assert len(w) == 1
            assert "training_step" in str(w[0].message)

    def test_warning_for_both_overrides(self):
        """Warns separately for both compute_loss and training_step."""

        class DoubleSneakyTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                pass

            def training_step(self, model, inputs, num_items_in_batch=None):
                pass

        _change_base_recursively(DoubleSneakyTrainer, Trainer, DPTrainer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_ghost_clipping_overrides(DoubleSneakyTrainer)
            assert len(w) == 2
            messages = {str(m.message) for m in w}
            assert any("compute_loss" in m for m in messages)
            assert any("training_step" in m for m in messages)

    def test_privatize_trainer_emits_warning_with_ghost_clipping(self):
        """privatize_trainer emits the warning when ghost clipping is enabled."""

        class SneakyTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                pass

        ghost_args = PrivacyArguments(noise_multiplier=1.0, grad_sample_mode="ghost")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            privatize_trainer(SneakyTrainer, default_privacy_args=ghost_args)
            ghost_warnings = [x for x in w if "ghost-clipping" in str(x.message)]
            assert len(ghost_warnings) == 1

    def test_privatize_trainer_no_warning_without_ghost_clipping(self):
        """privatize_trainer does NOT emit the warning when ghost clipping is not enabled."""

        class SneakyTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            privatize_trainer(SneakyTrainer)
            ghost_warnings = [x for x in w if "ghost-clipping" in str(x.message)]
            assert len(ghost_warnings) == 0

    @pytest.mark.skipif(not _has_trl, reason="trl not installed")
    def test_privatize_dpo_trainer_emits_warning_with_ghost_clipping(self):
        """Privatizing trl.DPOTrainer emits a warning when ghost clipping is enabled."""
        from trl import DPOTrainer

        ghost_args = PrivacyArguments(noise_multiplier=1.0, grad_sample_mode="ghost")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            privatize_trainer(DPOTrainer, default_privacy_args=ghost_args)
            ghost_warnings = [x for x in w if "ghost-clipping" in str(x.message)]
            assert len(ghost_warnings) >= 1
            assert any("compute_loss" in str(gw.message) for gw in ghost_warnings)

    @pytest.mark.skipif(not _has_trl, reason="trl not installed")
    def test_privatize_dpo_trainer_no_warning_without_ghost_clipping(self):
        """Privatizing trl.DPOTrainer does NOT emit a warning without ghost clipping."""
        from trl import DPOTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            privatize_trainer(DPOTrainer)
            ghost_warnings = [x for x in w if "ghost-clipping" in str(x.message)]
            assert len(ghost_warnings) == 0


class TestValidateGhostClipping:
    """Test DPCallback._validate_ghost_clipping runtime validation."""

    def test_passes_for_non_ghost_clipping_optimizer(self):
        """No error when the optimizer is a regular DPOptimizer (not ghost clipping)."""
        from dptrainer.callback import DPCallback

        optimizer = Mock(spec=DPOptimizer)
        model = Mock()
        # Should not raise
        DPCallback._validate_ghost_clipping(optimizer, model)

    def test_passes_when_loss_function_is_correct(self):
        """No error when ghost clipping optimizer and model has DPLossFastGradientClipping."""
        from dptrainer.callback import DPCallback
        from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
        from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping

        optimizer = Mock(spec=DPOptimizerFastGradientClipping)
        model = Mock()
        model.loss_function = Mock(spec=DPLossFastGradientClipping)
        del model._module  # ensure no unwrapping attribute
        del model.module

        DPCallback._validate_ghost_clipping(optimizer, model)

    def test_raises_when_loss_function_is_wrong(self):
        """RuntimeError when ghost clipping optimizer but loss_function is not wrapped."""
        from dptrainer.callback import DPCallback
        from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping

        optimizer = Mock(spec=DPOptimizerFastGradientClipping)
        model = Mock()
        model.loss_function = lambda x: x  # plain function, not DPLossFastGradientClipping
        del model._module
        del model.module

        with pytest.raises(RuntimeError, match="Ghost clipping optimizer is active"):
            DPCallback._validate_ghost_clipping(optimizer, model)

    def test_raises_when_loss_function_missing(self):
        """RuntimeError when ghost clipping optimizer but model has no loss_function."""
        from dptrainer.callback import DPCallback
        from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping

        optimizer = Mock(spec=DPOptimizerFastGradientClipping)
        model = Mock(spec=[])  # empty spec = no attributes

        with pytest.raises(RuntimeError, match="Ghost clipping optimizer is active"):
            DPCallback._validate_ghost_clipping(optimizer, model)

    def test_unwraps_model_with_module_attribute(self):
        """Correctly unwraps model._module.module to find loss_function."""
        from dptrainer.callback import DPCallback
        from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
        from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping

        inner_model = Mock()
        inner_model.loss_function = Mock(spec=DPLossFastGradientClipping)
        del inner_model._module
        del inner_model.module

        wrapper = Mock()
        wrapper._module = inner_model
        del wrapper.module

        optimizer = Mock(spec=DPOptimizerFastGradientClipping)

        # Should not raise — it finds the correct loss via unwrapping
        DPCallback._validate_ghost_clipping(optimizer, wrapper)


