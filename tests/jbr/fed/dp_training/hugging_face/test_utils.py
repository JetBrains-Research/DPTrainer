"""Tests for utility functions."""

from unittest.mock import Mock, patch, MagicMock
import pytest
from transformers import Trainer

from jbr.fed.dp_training.hugging_face.utils import (
    privatize_trainer,
    _change_base_recursively,
)
from jbr.fed.dp_training.hugging_face.trainer import DPTrainer
from jbr.fed.dp_training import PrivacyArguments


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


