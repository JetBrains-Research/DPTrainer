"""Tests for GradSampleHooks integration with DPTrainer.

These tests verify the new hooks-based non-wrapping mode from Opacus works
correctly with DPTrainer, including:
- prepare_module with wrap_model=False returns GradSampleHooks
- hooks.cleanup() properly removes hooks from model
- hooks.enable_hooks() / hooks.disable_hooks() work correctly
- Model is not wrapped (original model is used directly)
"""

import pytest
import torch
from torch import nn
from unittest.mock import patch, Mock

from opacus.grad_sample.utils import prepare_module
from opacus.grad_sample import GradSampleHooks

from dptrainer import PrivacyArguments
from dptrainer.trainer import DPTrainer


class TestPrepareModuleIntegration:
    """Test prepare_module with wrap_model=False as used by DPTrainer."""

    def test_prepare_module_returns_hooks_object(self, simple_model):
        """prepare_module with wrap_model=False returns a GradSampleHooks object."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        # Should return a hooks object, not wrap the model
        assert isinstance(hooks, GradSampleHooks)
        # Original model should be unchanged (not wrapped)
        assert isinstance(simple_model, nn.Module)
        assert not hasattr(simple_model, '_module')

    def test_prepare_module_hooks_have_cleanup_method(self, simple_model):
        """GradSampleHooks object has cleanup() method."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        assert hasattr(hooks, 'cleanup')
        assert callable(hooks.cleanup)

    def test_prepare_module_hooks_have_enable_disable(self, simple_model):
        """GradSampleHooks object has enable_hooks() and disable_hooks() methods."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        assert hasattr(hooks, 'enable_hooks')
        assert hasattr(hooks, 'disable_hooks')
        assert callable(hooks.enable_hooks)
        assert callable(hooks.disable_hooks)


class TestHooksCleanup:
    """Test that hooks cleanup works correctly."""

    def test_cleanup_removes_hooks(self, simple_model):
        """cleanup() should remove all registered hooks from the model."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        # Count hooks before cleanup
        initial_forward_hooks = sum(
            len(m._forward_hooks) for m in simple_model.modules()
        )
        initial_backward_hooks = sum(
            len(m._backward_hooks) for m in simple_model.modules()
        )

        # There should be some hooks registered
        assert initial_forward_hooks > 0 or initial_backward_hooks > 0

        # Cleanup
        hooks.cleanup()

        # Count hooks after cleanup
        final_forward_hooks = sum(
            len(m._forward_hooks) for m in simple_model.modules()
        )
        final_backward_hooks = sum(
            len(m._backward_hooks) for m in simple_model.modules()
        )

        # All hooks should be removed
        assert final_forward_hooks == 0
        assert final_backward_hooks == 0

    def test_model_still_usable_after_cleanup(self, simple_model):
        """Model should still work normally after cleanup."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        hooks.cleanup()

        # Model should still be usable for inference
        x = torch.randn(1, 10)
        output = simple_model(x)

        assert output is not None
        assert output.shape == (1, 2)


class TestHooksEnableDisable:
    """Test hooks enable/disable functionality."""

    def test_disable_hooks_prevents_grad_sampling(self, simple_model):
        """disable_hooks() should prevent per-sample gradient computation."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        # Disable hooks
        hooks.disable_hooks()

        # Forward pass
        x = torch.randn(2, 10)
        output = simple_model(x)
        loss = output.sum()
        loss.backward()

        # With hooks disabled, grad_sample should not be computed
        for param in simple_model.parameters():
            if param.requires_grad:
                # Should have regular gradients
                assert param.grad is not None
                # Should NOT have per-sample gradients
                assert not hasattr(param, 'grad_sample') or param.grad_sample is None

    def test_enable_hooks_after_disable(self, simple_model):
        """enable_hooks() should restore per-sample gradient computation."""
        hooks = prepare_module(
            simple_model,
            grad_sample_mode="hooks",
            wrap_model=False,
        )

        # Disable then re-enable
        hooks.disable_hooks()
        hooks.enable_hooks()

        # Forward pass
        x = torch.randn(2, 10)
        simple_model.zero_grad()
        output = simple_model(x)
        loss = output.sum()
        loss.backward()

        # With hooks re-enabled, grad_sample should be computed
        found_grad_sample = False
        for param in simple_model.parameters():
            if param.requires_grad and hasattr(param, 'grad_sample') and param.grad_sample is not None:
                found_grad_sample = True
                # Per-sample gradients should have batch dimension
                assert param.grad_sample.shape[0] == 2  # batch size

        assert found_grad_sample, "Expected per-sample gradients after re-enabling hooks"


class TestDPTrainerHooksIntegration:
    """Test DPTrainer correctly uses hooks-based mode."""

    @patch("dptrainer.trainer.prepare_module")
    def test_dptrainer_uses_wrap_model_false(
        self, mock_prepare_module, simple_model, small_dataset, training_args
    ):
        """DPTrainer should call prepare_module with wrap_model=False."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Verify prepare_module was called with wrap_model=False
        mock_prepare_module.assert_called_once()
        call_kwargs = mock_prepare_module.call_args[1]
        assert call_kwargs.get('wrap_model') is False

    @patch("dptrainer.trainer.prepare_module")
    def test_dptrainer_stores_hooks_reference(
        self, mock_prepare_module, simple_model, small_dataset, training_args
    ):
        """DPTrainer should store the hooks object for later cleanup."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Trainer should have hooks attribute
        assert hasattr(trainer, 'hooks')
        assert trainer.hooks is mock_hooks

    @patch("dptrainer.trainer.prepare_module")
    def test_detach_model_calls_cleanup(
        self, mock_prepare_module, simple_model, small_dataset, training_args
    ):
        """detach_model() should call hooks.cleanup()."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Detach model
        detached = trainer.detach_model()

        # Cleanup should have been called
        mock_hooks.cleanup.assert_called_once()
        # Should return the original model
        assert detached is simple_model


class TestModelNotWrapped:
    """Test that the model is not wrapped in hooks-based mode."""

    @patch("dptrainer.trainer.prepare_module")
    def test_model_remains_original_type(
        self, mock_prepare_module, simple_model, small_dataset, training_args
    ):
        """Model should remain its original type, not wrapped in GradSampleModule."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        original_type = type(simple_model)

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Model type should be unchanged
        assert type(trainer.model) == original_type
        # Model should not have _module attribute (sign of wrapping)
        assert not hasattr(trainer.model, '_module')

    @patch("dptrainer.trainer.prepare_module")
    def test_model_forward_works_directly(
        self, mock_prepare_module, simple_model, small_dataset, training_args
    ):
        """Forward pass should work directly on the model."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Forward pass should work directly
        # Get device from model parameters
        device = next(trainer.model.parameters()).device
        x = torch.randn(1, 10, device=device)
        output = trainer.model(x)

        assert output is not None
        assert output.shape == (1, 2)


class TestGhostClippingHooks:
    """Test ghost clipping (fast gradient clipping) mode with hooks."""

    @patch("dptrainer.trainer.set_loss_function")
    @patch("dptrainer.trainer.DPLossFastGradientClipping")
    @patch("dptrainer.trainer.prepare_module")
    def test_ghost_mode_uses_correct_hooks_class(
        self, mock_prepare_module, mock_dp_loss, mock_set_loss, simple_model, small_dataset, training_args
    ):
        """Ghost clipping mode should use the appropriate hooks class."""
        mock_hooks = Mock()
        mock_hooks.cleanup = Mock()
        mock_prepare_module.return_value = mock_hooks

        # Add loss_function to model for ghost clipping
        loss_fn = nn.CrossEntropyLoss()
        loss_fn.reduction = "mean"
        simple_model.loss_function = loss_fn

        privacy_args = PrivacyArguments(
            noise_multiplier=1.0,
            grad_sample_mode="ghost",
        )

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        # Verify prepare_module was called with ghost mode
        mock_prepare_module.assert_called_once()
        call_kwargs = mock_prepare_module.call_args[1]
        assert call_kwargs.get('grad_sample_mode') == 'ghost'
        assert call_kwargs.get('wrap_model') is False

        # Verify DPLossFastGradientClipping was created for ghost mode
        mock_dp_loss.assert_called_once()
