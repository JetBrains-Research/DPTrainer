import logging
import warnings

from opacus.accountants import create_accountant
from opacus.optimizers import DPOptimizer
from riskcal import CTDAccountant
from riskcal.conversions import get_beta_from_pld, get_advantage_from_pld
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
        target_alpha: float,
        max_epsilon: float = None,
        min_beta: float = None,
    ) -> None:
        """Initialize the DPCallback.

        Args:
            accountant (str): The privacy accountant mechanism to use (e.g., "rdp", "ctd").
            gradient_accumulation_steps (int): Number of gradient accumulation steps.
            target_delta (float): Target delta for (epsilon, delta)-DP.
            target_alpha (float): Target false positive rate for trade-off function accounting.
            max_epsilon (float): Maximum allowed epsilon before stopping training.
            min_beta (float): Minimum allowed beta before stopping training.
        """
        self.accountant = create_accountant(accountant)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_delta = target_delta
        self.target_alpha = target_alpha
        self.max_epsilon = max_epsilon
        self.min_beta = min_beta

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

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        """Clean up extra elements in the optimizer step skip queue at the beginning of each step."""
        optimizer = self._get_dp_optimizer(optimizer)

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
            dict: Dictionary containing privacy metrics (e.g., privacy_epsilon, privacy_beta, privacy_advantage).
        """
        metrics = {}
        if isinstance(self.accountant, CTDAccountant):
            pld = self.accountant.get_pld(grid_step=0.01) if len(self.accountant.history) > 0 else None
            if self.target_delta is not None:
                metrics["privacy_epsilon"] = pld.get_epsilon_for_delta(delta=self.target_delta) if pld else 0.0
            if self.target_alpha is not None:
                metrics["privacy_beta"] = get_beta_from_pld(pld, self.target_alpha) if pld else 1.0
            metrics["privacy_advantage"] = get_advantage_from_pld(pld) if pld else 0.0
        else:
            if self.target_delta is not None:
                with warnings.catch_warnings(category=UserWarning, action="ignore"):
                    metrics["privacy_epsilon"] = (self.accountant.get_epsilon(self.target_delta)
                                                  if len(self.accountant.history) > 0 else 0.0)

        return metrics


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

        if "privacy_beta" in metrics and self.min_beta is not None and metrics["privacy_beta"] < self.min_beta:
            logger.warning(f"Min beta exceeded: {metrics['privacy_beta']} < {self.min_beta}. Stopping training...")
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
                "target_alpha": self.target_alpha,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_epsilon": self.max_epsilon,
                "min_beta": self.min_beta,
            }, "attributes": {
                "_accountant_state_dict": self._accountant_state_dict,
            }
        }
