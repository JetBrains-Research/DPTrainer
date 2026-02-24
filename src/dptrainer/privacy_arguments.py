import warnings
from dataclasses import dataclass, field
from typing import Optional

from opacus.accountants.utils import get_noise_multiplier
from transformers import logging

logger = logging.get_logger(__name__)

@dataclass
class PrivacyArguments:
    """Dataclass for all privacy-related training parameters.

    Attributes:
        accountant (str): Accountant mechanism to use for DP training.
        grad_sample_mode (str): Grad sample mode of Opacus.
        per_sample_max_grad_norm (float): Max per sample clip norm.
        clipping (str): Clipping strategy.
        poisson_sampling (bool): Use Poisson sampling, use standard batches otherwise.
        min_clipbound (float): Min clip bound of the AdaClip algorithm.
        max_clipbound (float): Max clip bound of the AdaClip algorithm.
        clipbound_learning_rate (float): Learning rate of the AdaClip algorithm.
        target_unclipped_quantile (float): Target fraction of unclipped samples per batch of the AdaClip algorithm.
        unclipped_num_std (float): Standard deviation of the unclipped number noise of the AdaClip algorithm.
        noise_multiplier (Optional[float]): Noise multiplier for DP training.
        target_epsilon (Optional[float]): Target epsilon at end of training (mutually exclusive with noise multiplier).
        target_delta (Optional[float]): Target delta, defaults to 1/N.
    """
    accountant: str = field(default="rdp", metadata={"help": "Accountant mechanism to use for DP training"})
    grad_sample_mode: str = field(default="hooks", metadata={"help": "Grad sample mode of Opacus"})
    per_sample_max_grad_norm: float = field(default=0.5, metadata={"help": "Max per sample clip norm"})
    clipping: str = field(default="flat", metadata={"help": "Clipping strategy"})
    poisson_sampling: bool = field(default=True, metadata={"help": "Use Poisson sampling, use standard batches otherwise"})
    min_clipbound: float = field(default=0.05, metadata={"help": "Min clip bound of the AdaClip algorithm"})
    max_clipbound: float = field(default=1e8, metadata={"help": "Max clip bound of the AdaClip algorithm"})
    clipbound_learning_rate: float = field(default=0.2, metadata={"help": "Learning rate of the AdaClip algorithm"})
    target_unclipped_quantile: float = field(default=0.5, metadata={
        "help": "Target fraction of unclipped samples per batch of the AdaClip algorithm"})
    unclipped_num_std: float = field(default=1.0, metadata={
        "help": "Standard deviation of the unclipped number noise of the AdaClip algorithm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=None, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"})
    target_delta: Optional[float] = field(default=None, metadata={"help": "Target delta, defaults to 1/N"})

    @classmethod
    def low_privacy(cls):
        """Create a low-privacy configuration with no noise and no Poisson sampling.

        Returns:
            PrivacyArguments: A PrivacyArguments instance with minimal privacy guarantees.
        """
        return cls(accountant="rdp", noise_multiplier=0.0, poisson_sampling=False)

    def precalculate(self, num_samples: int, sample_rate: float, steps: int):
        """Precalculate the noise multiplier based on target privacy parameters.

        Sets `target_delta` to 1/N if not provided, then computes `noise_multiplier`
        from `target_epsilon` if `noise_multiplier` is not already set.

        Args:
            num_samples (int): Total number of samples in the training dataset.
            sample_rate (float): The sampling rate of each batch.
            steps (int): Total number of training steps.
        """
        if self.target_delta is None:
            self.target_delta = 1.0 / num_samples

        if self.noise_multiplier is not None:
            return

        if self.target_epsilon is not None:
            with warnings.catch_warnings(category=UserWarning, action="ignore"):
                self.noise_multiplier = get_noise_multiplier(target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta, sample_rate=sample_rate, steps=steps, accountant=self.accountant)
        else:
            self.noise_multiplier = 0.0
