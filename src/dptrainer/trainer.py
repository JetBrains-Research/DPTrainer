import math
import warnings
from typing import Optional, Union, Callable

import datasets
import torch
from opacus.grad_sample.utils import prepare_module
from opacus.data_loader import DPDataLoader
from opacus.optimizers import get_optimizer_class, AdaClipDPOptimizer
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import nn
from transformers import (
    Trainer, logging, modeling_utils, TrainingArguments, PreTrainedModel, TrainerCallback
)
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from dptrainer.utils import set_loss_function
from dptrainer.privacy_arguments import PrivacyArguments
from dptrainer.callback import DPCallback

logger = logging.get_logger(__name__)


class DPTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            train_dataset: Union[datasets.Dataset, torch.utils.data.Dataset] = None,
            privacy_args: PrivacyArguments = None,
            compute_metrics: Optional[Callable] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            **kwargs
    ):
        """Hugging Face Trainer with Differential Privacy support.

        Args:
            model (Union[PreTrainedModel, nn.Module]): Model to train.
            args (TrainingArguments): Training arguments.
            train_dataset (Union[datasets.Dataset, torch.utils.data.Dataset]): Training dataset.
            privacy_args (PrivacyArguments): Privacy arguments for differential private training.
            compute_metrics (Optional[Callable]): Custom evaluation metrics.
            callbacks (Optional[list[TrainerCallback]]): Training callbacks.
            **kwargs: Additional keyword arguments passed to Trainer.
        """
        self.privacy_args = privacy_args or getattr(self, "default_privacy_args", None)
        if self.privacy_args is None:
            raise ValueError("Privacy arguments must be provided.")

        dataset_size = len(train_dataset)
        if (isinstance(train_dataset, (datasets.IterableDataset, torch.utils.data.IterableDataset))
                and privacy_args.poisson_sampling):
            raise ValueError("IterableDataset is not supported by DPTrainer when poisson_sampling is True.")
        if self.privacy_args.grad_sample_mode.startswith("ew"):
            raise ValueError("Expanded Weights cannot be used with DPTrainer.")
        if self.privacy_args.poisson_sampling:
            if args.dataloader_num_workers > 1:
                raise ValueError("Poisson sampling requires dataloader_num_workers to be 0 or 1.")
        if args.save_strategy and args.save_steps and not args.restore_callback_states_from_checkpoint:
            warnings.warn("Save strategy is set but restore_callback_states_from_checkpoint is false. "
                          "Accountant states will not be restored from the checkpoint leading to the incorrect "
                          "privacy budget estimates. Setting restore_callback_states_from_checkpoint to True")
            args.restore_callback_states_from_checkpoint = True
        if args.world_size > 1:
            raise ValueError("Distributed training is not supported by DPTrainer.")

        sample_rate = args.per_device_train_batch_size * args.gradient_accumulation_steps / dataset_size

        self.privacy_args.precalculate(
            num_samples=dataset_size,
            sample_rate=sample_rate,
            steps=(args.max_steps // args.gradient_accumulation_steps if args.max_steps and args.max_steps != -1
                   else math.ceil(1 / sample_rate) * args.num_train_epochs)
        )

        logger.info(f"Using privacy noise multiplier: {self.privacy_args.noise_multiplier}")

        # Remove existing hooks if present (e.g., when reusing models in tests)
        if hasattr(model, "autograd_grad_sample_hooks"):
            while model.autograd_grad_sample_hooks:
                handle = model.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(model, "autograd_grad_sample_hooks")

        self.hooks = prepare_module(
            model,
            grad_sample_mode=self.privacy_args.grad_sample_mode,
            wrap_model=False,
        )

        dp_callback = DPCallback(
            accountant=self.privacy_args.accountant,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            target_delta=self.privacy_args.target_delta,
            max_epsilon=self.privacy_args.target_epsilon,
        )
        callbacks = callbacks or []
        callbacks.append(dp_callback)

        def compute_privacy_metrics(*args, compute_result: bool = True, **kwargs):
            if compute_metrics:
                metrics = compute_metrics(*args, compute_result, **kwargs) or {}
            else:
                metrics = {}

            if compute_result:
                privacy_metrics = dp_callback.get_privacy_metrics()
                metrics.update(privacy_metrics)

            return metrics

        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks,
                         compute_metrics=compute_privacy_metrics, **kwargs)

        optimizer = self.create_optimizer()
        optimizer.attach_step_hook(dp_callback.get_optimizer_callback(sample_rate=sample_rate))

        if self.privacy_args and self.privacy_args.grad_sample_mode == "ghost":
            criterion = self.model.loss_function
            if not hasattr(criterion, "reduction"):
                setattr(criterion, "reduction", "mean")
            criterion = DPLossFastGradientClipping(self.hooks,
                                                   self.create_optimizer(),
                                                   criterion,
                                                   loss_reduction="mean")
            set_loss_function(model, criterion)

    def create_optimizer(self):
        """Create a differentially private optimizer wrapping the base optimizer.

        Returns:
            DPOptimizer: The differentially private optimizer.
        """
        if self.optimizer:
            return self.optimizer

        self.optimizer = super().create_optimizer()

        optim_class = get_optimizer_class(
            clipping=self.privacy_args.clipping \
                if not (self.privacy_args.clipping == "adaptive" and
                        self.privacy_args.grad_sample_mode == "ghost") \
                else "flat",  # flat, adaptive, per_layer
            distributed=False,
            grad_sample_mode=self.privacy_args.grad_sample_mode,
        )

        if issubclass(optim_class, AdaClipDPOptimizer):
            self.optimizer = optim_class(
                optimizer=self.optimizer,
                noise_multiplier=self.privacy_args.noise_multiplier,
                expected_batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
                max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                max_clipbound=self.privacy_args.max_clipbound,
                min_clipbound=self.privacy_args.min_clipbound,
                clipbound_learning_rate=self.privacy_args.clipbound_learning_rate,
                target_unclipped_quantile=self.privacy_args.target_unclipped_quantile,
                unclipped_num_std=self.privacy_args.unclipped_num_std,
                loss_reduction="mean",
            )
        else:
            self.optimizer = optim_class(
                optimizer=self.optimizer,
                noise_multiplier=self.privacy_args.noise_multiplier,
                max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                expected_batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
                loss_reduction="mean",
            )

        return self.optimizer

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create a training dataloader with privacy-compatible batching.

        Returns:
            torch.utils.data.DataLoader: The training dataloader with optional Poisson sampling and batch memory management.
        """
        data_loader = self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

        if self.privacy_args.poisson_sampling:
            data_loader = DPDataLoader.from_data_loader(data_loader, rand_on_empty=True)

        # Get the optimizer, unwrapping AcceleratedOptimizer if present
        optimizer = self.create_optimizer()
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer

        data_loader = wrap_data_loader(
            data_loader=data_loader,
            optimizer=optimizer,
            max_batch_size=self._train_batch_size
        )

        return data_loader

    def detach_model(self) -> nn.Module:
        """Detach the model from the hooks and return the model.

        The method cleans up resources or connections associated with the private trainer
        and detaches the managed model for further usage, if needed.

        Returns:
            nn.Module: Detached model.
        """
        self.hooks.cleanup()

        return self.model