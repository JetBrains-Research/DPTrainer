import logging
import warnings

from opacus.accountants import create_accountant
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from transformers import TrainerCallback, TrainerControl, TrainingArguments, TrainerState
from transformers.trainer_callback import ExportableState

logger = logging.getLogger(__name__)


class DPCallback(TrainerCallback, ExportableState):
    """
    Trainer callback that makes `transformers.Trainer` compatible with Opacus.

    Handles privacy accounting, budget enforcement, and optimizer hooks.
    """
    def __init__(
        self,
        accountant: str,
        gradient_accumulation_steps: int,
        target_delta: float,
        max_epsilon: float = None,
    ) -> None:
        """Initialize the DPCallback.

        Args:
            accountant (str): The privacy accountant mechanism to use (e.g., "rdp").
            gradient_accumulation_steps (int): Number of gradient accumulation steps.
            target_delta (float): Target delta for (epsilon, delta)-DP.
            max_epsilon (float): Maximum allowed epsilon before stopping training.
        """
        self.accountant = create_accountant(accountant)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_delta = target_delta
        self.max_epsilon = max_epsilon
        self._ghost_clipping_validated = False

    def get_optimizer_callback(self, sample_rate):
        """Get the optimizer hook function for privacy accounting.

        Args:
            sample_rate: The sampling rate of the data loader.

        Returns:
            Callable: The optimizer hook function.
        """
        return self.accountant.get_optimizer_hook_fn(sample_rate)

    def on_train_begin(self, args, state, control, **kwargs):
        """Check if the privacy budget is already exceeded at the start of training."""
        return self._check_max_privacy_budget_exceeded(control)

    def on_step_begin(self, args, state, control, optimizer=None, model=None, **kwargs):
        """Clean up extra elements in the optimizer step skip queue at the beginning of each step."""
        optimizer = self._get_dp_optimizer(optimizer)

        if not self._ghost_clipping_validated:
            self._validate_ghost_clipping(optimizer, model)
            self._ghost_clipping_validated = True

        # trainer samples one extra element at the beginning of each epoch, cleaning it up if present
        while len(optimizer._step_skip_queue) > self.gradient_accumulation_steps:
            optimizer._step_skip_queue.pop(0)


    def on_substep_end(self, args, state, control, optimizer=None, **kwargs):
        """Step the optimizer and clear gradients after each gradient accumulation substep."""
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        """Clear gradients after each training step."""
        optimizer = self._get_dp_optimizer(optimizer)

        # gradients should be cleared after each substep with poisson sampling
        # precalculated grad_sample will stay until the final aggregation
        # optimizer.step() is executed by the trainer
        optimizer.zero_grad(set_to_none=True)

    def on_evaluate(self, args, state, control, optimizer=None, metrics=None, **kwargs):
        """Check if the privacy budget is exceeded after evaluation."""
        return self._check_max_privacy_budget_exceeded(control)

    def get_privacy_metrics(self):
        """Compute current privacy metrics from the accountant.

        Returns:
            dict: Dictionary containing privacy metrics (e.g., privacy_epsilon).
        """
        metrics = {}
        if self.target_delta is not None:
            with warnings.catch_warnings(category=UserWarning, action="ignore"):
                metrics["privacy_epsilon"] = (self.accountant.get_epsilon(self.target_delta)
                                              if len(self.accountant.history) > 0 else 0.0)

        return metrics


    @staticmethod
    def _validate_ghost_clipping(optimizer, model):
        """Validate that ghost clipping is properly configured when using a ghost clipping optimizer.

        Checks that the model's loss_function is wrapped with DPLossFastGradientClipping
        when using a ghost clipping optimizer, ensuring privacy gradients are computed correctly.

        Args:
            optimizer: The DP optimizer (possibly wrapped).
            model: The model being trained.

        Raises:
            RuntimeError: If ghost clipping optimizer is used but the model's loss_function
                is not wrapped with DPLossFastGradientClipping.
        """
        # Unwrap to the actual DP optimizer
        dp_optimizer = optimizer
        for _ in range(10):
            if isinstance(dp_optimizer, DPOptimizer):
                break
            elif hasattr(dp_optimizer, 'optimizer'):
                dp_optimizer = dp_optimizer.optimizer
            elif hasattr(dp_optimizer, '_optimizer'):
                dp_optimizer = dp_optimizer._optimizer
            else:
                return

        if not isinstance(dp_optimizer, DPOptimizerFastGradientClipping):
            return

        # Ghost clipping optimizer detected — verify the loss function is properly wrapped
        unwrapped = model
        if hasattr(unwrapped, '_module'):
            unwrapped = unwrapped._module
        if hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module

        loss_fn = getattr(unwrapped, 'loss_function', None)
        if not isinstance(loss_fn, DPLossFastGradientClipping):
            raise RuntimeError(
                f"Ghost clipping optimizer is active but the model's loss_function "
                f"is {type(loss_fn).__name__}, not DPLossFastGradientClipping. "
                f"This means ghost clipping was bypassed — likely because a custom "
                f"trainer or model overrides compute_loss or training_step, or "
                f"replaces the loss_function after DPTrainer initialization. "
                f"Privacy gradients will NOT be computed correctly."
            )

    def _get_dp_optimizer(self, optimizer) -> DPOptimizer:

        for i in range(10):
            if isinstance(optimizer, DPOptimizer):
                return optimizer
            elif hasattr(optimizer, 'optimizer'):  # accelerate.Optimizer
                optimizer = optimizer.optimizer
            elif hasattr(optimizer, "_optimizer"):
                optimizer = optimizer._optimizer
            else:
                break

        raise ValueError(f"Expected DPOptimizer, got {type(optimizer)}")

    def _check_max_privacy_budget_exceeded(self, control: TrainerControl) -> TrainerControl:
        metrics = self.get_privacy_metrics()
        if ("privacy_epsilon" in metrics and self.max_epsilon is not None
                and metrics["privacy_epsilon"] >= self.max_epsilon):
            logger.warning(f"Max epsilon exceeded: {metrics['privacy_epsilon']} >= {self.max_epsilon}." 
                           "Stopping training...")
            control.should_training_stop = True

        return control

    @property
    def _accountant_state_dict(self):
        state_dict = self.accountant.state_dict()

        return state_dict

    @_accountant_state_dict.setter
    def _accountant_state_dict(self, state_dict):
        self.accountant.load_state_dict(state_dict)

    def state(self) -> dict:
        """Return the exportable state of the callback for checkpointing.

        Returns:
            dict: Dictionary containing the callback's constructor args and accountant state.
        """
        return {
            "args": {
                "accountant": self.accountant.mechanism(),
                "target_delta": self.target_delta,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_epsilon": self.max_epsilon,
            }, "attributes": {
                "_accountant_state_dict": self._accountant_state_dict,
            }
        }
