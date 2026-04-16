"""Tests for DPCallback."""

from unittest.mock import Mock, MagicMock, patch
import pytest
from transformers import TrainerControl

from dptrainer.callback import DPCallback


class TestDPCallbackInitialization:
    """Test DPCallback initialization."""

    def test_initialization_basic(self):
        """Test basic initialization with required parameters."""
        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )
        assert callback.target_delta == 1e-5
        assert callback.gradient_accumulation_steps == 2
        assert callback.max_epsilon is None

    def test_initialization_with_privacy_budget_limits(self):
        """Test initialization with max_epsilon."""
        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
            max_epsilon=3.0,
        )
        assert callback.max_epsilon == 3.0


class TestDPCallbackPrivacyMetrics:
    """Test privacy metrics calculation."""

    @patch("dptrainer.callback.create_accountant")
    def test_get_privacy_metrics_rdp_empty_history(self, mock_create_accountant):
        """Test privacy metrics with RDP accountant and empty history."""
        mock_accountant = Mock()
        mock_accountant.history = []
        mock_accountant.get_epsilon = Mock(return_value=0.0)
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        metrics = callback.get_privacy_metrics()
        assert "privacy_epsilon" in metrics
        assert metrics["privacy_epsilon"] == 0.0

    @patch("dptrainer.callback.create_accountant")
    def test_get_privacy_metrics_rdp_with_history(self, mock_create_accountant):
        """Test privacy metrics with RDP accountant and non-empty history."""
        mock_accountant = Mock()
        mock_accountant.history = [1]  # Non-empty
        mock_accountant.get_epsilon = Mock(return_value=2.5)
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        metrics = callback.get_privacy_metrics()
        assert "privacy_epsilon" in metrics
        assert metrics["privacy_epsilon"] == 2.5
        mock_accountant.get_epsilon.assert_called_once_with(1e-5)

    @patch("dptrainer.callback.create_accountant")
    def test_get_privacy_metrics_no_target_delta(self, mock_create_accountant):
        """Test privacy metrics when target_delta is None."""
        mock_accountant = Mock()
        mock_accountant.history = []
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=None,  # No target_delta
        )

        metrics = callback.get_privacy_metrics()
        # Should return empty or minimal metrics when no targets set
        assert isinstance(metrics, dict)


class TestDPCallbackGetDPOptimizer:
    """Test _get_dp_optimizer unwrapping logic."""

    def test_get_dp_optimizer_direct(self):
        """Test getting DP optimizer when passed directly."""
        from opacus.optimizers import DPOptimizer

        mock_optimizer = Mock(spec=DPOptimizer)

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        result = callback._get_dp_optimizer(mock_optimizer)
        assert result is mock_optimizer

    def test_get_dp_optimizer_wrapped_once(self):
        """Test getting DP optimizer wrapped in another optimizer."""
        from opacus.optimizers import DPOptimizer

        mock_dp_optimizer = Mock(spec=DPOptimizer)
        mock_wrapper = Mock()
        mock_wrapper.optimizer = mock_dp_optimizer

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        result = callback._get_dp_optimizer(mock_wrapper)
        assert result is mock_dp_optimizer

    def test_get_dp_optimizer_not_found(self):
        """Test error when DP optimizer is not found."""
        mock_optimizer = Mock()
        # No .optimizer or ._optimizer attribute

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        with pytest.raises(ValueError, match="Expected DPOptimizer"):
            callback._get_dp_optimizer(mock_optimizer)


class TestDPCallbackPrivacyBudgetChecks:
    """Test privacy budget exceeded checks."""

    @patch("dptrainer.callback.create_accountant")
    def test_max_epsilon_exceeded(self, mock_create_accountant):
        """Test training stops when max_epsilon is exceeded."""
        mock_accountant = Mock()
        mock_accountant.history = [1]
        mock_accountant.get_epsilon = Mock(return_value=5.0)
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
            max_epsilon=3.0,
        )

        control = TrainerControl()
        result = callback._check_max_privacy_budget_exceeded(control)

        assert result.should_training_stop is True

    @patch("dptrainer.callback.create_accountant")
    def test_max_epsilon_not_exceeded(self, mock_create_accountant):
        """Test training continues when max_epsilon is not exceeded."""
        mock_accountant = Mock()
        mock_accountant.history = [1]
        mock_accountant.get_epsilon = Mock(return_value=2.0)
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
            max_epsilon=3.0,
        )

        control = TrainerControl()
        result = callback._check_max_privacy_budget_exceeded(control)

        assert result.should_training_stop is False


class TestDPCallbackState:
    """Test callback state serialization."""

    @patch("dptrainer.callback.create_accountant")
    def test_state_serialization(self, mock_create_accountant):
        """Test that state() returns proper structure."""
        mock_accountant = Mock()
        mock_accountant.mechanism = Mock(return_value="rdp")
        mock_accountant.state_dict = Mock(return_value={"step": 10})
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
            max_epsilon=3.0,
        )

        state = callback.state()

        assert "args" in state
        assert "attributes" in state
        assert state["args"]["accountant"] == "rdp"
        assert state["args"]["target_delta"] == 1e-5
        assert state["args"]["gradient_accumulation_steps"] == 2
        assert state["args"]["max_epsilon"] == 3.0
        assert "_accountant_state_dict" in state["attributes"]

    @patch("dptrainer.callback.create_accountant")
    def test_accountant_state_dict_setter(self, mock_create_accountant):
        """Test that accountant state dict can be set."""
        mock_accountant = Mock()
        mock_accountant.mechanism = Mock(return_value="rdp")
        mock_accountant.state_dict = Mock(return_value={})
        mock_accountant.load_state_dict = Mock()
        mock_create_accountant.return_value = mock_accountant

        callback = DPCallback(
            accountant="rdp",
            gradient_accumulation_steps=2,
            target_delta=1e-5,
        )

        test_state = {"step": 5, "history": [1, 2, 3]}
        callback._accountant_state_dict = test_state

        mock_accountant.load_state_dict.assert_called_once_with(test_state)
