"""Tests for DPTrainer."""

from unittest.mock import Mock, call, patch, MagicMock
import pytest
import torch
from torch.utils.data import IterableDataset

from dptrainer import PrivacyArguments
from dptrainer.trainer import DPTrainer


class TestDPTrainerInitialization:
    """Test DPTrainer initialization."""

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_success(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test successful initialization with valid arguments."""
        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        assert trainer.privacy_args is privacy_args
        assert trainer.hooks is mock_controller
        mock_wrap_model.assert_called_once()

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_without_privacy_args(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test that initialization fails without privacy_args."""
        with pytest.raises(ValueError, match="Privacy arguments must be provided"):
            DPTrainer(
                model=simple_model,
                args=training_args,
                train_dataset=small_dataset,
                privacy_args=None,
            )

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_iterable_dataset_with_poisson(
        self, mock_wrap_model, simple_model, training_args
    ):
        """Test that IterableDataset with poisson_sampling raises error."""

        class SimpleIterableDataset(IterableDataset):
            def __len__(self):
                return 10

            def __iter__(self):
                for i in range(10):
                    yield {"input": torch.randn(5), "label": i % 2}

        iterable_dataset = SimpleIterableDataset()
        privacy_args = PrivacyArguments(noise_multiplier=1.0, poisson_sampling=True)

        with pytest.raises(
            ValueError,
            match="IterableDataset is not supported by DPTrainer when poisson_sampling is True",
        ):
            DPTrainer(
                model=simple_model,
                args=training_args,
                train_dataset=iterable_dataset,
                privacy_args=privacy_args,
            )

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_expanded_weights_rejected(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test that expanded weights mode is rejected."""
        privacy_args = PrivacyArguments(
            noise_multiplier=1.0, grad_sample_mode="ew_clipping"
        )

        with pytest.raises(
            ValueError, match="Expanded Weights cannot be used with DPTrainer"
        ):
            DPTrainer(
                model=simple_model,
                args=training_args,
                train_dataset=small_dataset,
                privacy_args=privacy_args,
            )

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_enables_checkpoint_restoration(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test that checkpoint restoration is enabled when save_strategy is set."""
        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        training_args.save_strategy = "steps"
        training_args.save_steps = 10
        training_args.restore_callback_states_from_checkpoint = False

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        with pytest.warns(UserWarning, match="restore_callback_states_from_checkpoint"):
            trainer = DPTrainer(
                model=simple_model,
                args=training_args,
                train_dataset=small_dataset,
                privacy_args=privacy_args,
            )

        assert training_args.restore_callback_states_from_checkpoint is True

    @patch("dptrainer.trainer.prepare_module")
    def test_initialization_distributed_training_rejected(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test that distributed training is rejected."""
        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        with patch.object(type(training_args), "world_size", new_callable=lambda: property(lambda self: 2)):
            with pytest.raises(
                ValueError, match="Distributed training is not supported by DPTrainer"
            ):
                DPTrainer(
                    model=simple_model,
                    args=training_args,
                    train_dataset=small_dataset,
                    privacy_args=privacy_args,
                )


class TestDPTrainerCreateOptimizer:
    """Test optimizer creation."""

    @patch("dptrainer.trainer.prepare_module")
    @patch("dptrainer.trainer.get_optimizer_class")
    def test_create_optimizer_flat_clipping(
        self,
        mock_get_optimizer_class,
        mock_wrap_model,
        simple_model,
        small_dataset,
        training_args,
    ):
        """Test optimizer creation with flat clipping."""
        from opacus.optimizers import DPOptimizer

        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        # Return actual DPOptimizer class (not AdaClip)
        mock_get_optimizer_class.return_value = DPOptimizer

        privacy_args = PrivacyArguments(
            noise_multiplier=1.0, clipping="flat", per_sample_max_grad_norm=1.0
        )

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        optimizer = trainer.create_optimizer()

        # Verify get_optimizer_class was called correctly
        mock_get_optimizer_class.assert_called_once_with(
            clipping="flat", distributed=False, grad_sample_mode="hooks"
        )
        # Verify optimizer was created and is the right type
        assert optimizer is not None
        assert isinstance(optimizer, DPOptimizer)

    @patch("dptrainer.trainer.prepare_module")
    @patch("dptrainer.trainer.get_optimizer_class")
    def test_create_optimizer_adaptive_clipping(
        self,
        mock_get_optimizer_class,
        mock_wrap_model,
        simple_model,
        small_dataset,
        training_args,
    ):
        """Test optimizer creation with adaptive clipping."""
        from opacus.optimizers import AdaClipDPOptimizer

        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        # Return the actual AdaClipDPOptimizer class
        mock_get_optimizer_class.return_value = AdaClipDPOptimizer

        privacy_args = PrivacyArguments(
            noise_multiplier=1.0,
            clipping="adaptive",
            per_sample_max_grad_norm=1.0,
            min_clipbound=0.1,
            max_clipbound=10.0,
            clipbound_learning_rate=0.2,
            target_unclipped_quantile=0.5,
        )

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        optimizer = trainer.create_optimizer()

        # Verify optimizer was created and is the right type
        assert optimizer is not None
        assert isinstance(optimizer, AdaClipDPOptimizer)


class TestDPTrainerGetTrainDataloader:
    """Test train dataloader creation."""

    @patch("dptrainer.trainer.prepare_module")
    @patch("dptrainer.trainer.wrap_data_loader")
    @patch("dptrainer.trainer.DPDataLoader")
    def test_get_train_dataloader_with_poisson(
        self,
        mock_dp_dataloader,
        mock_wrap_data_loader,
        mock_wrap_model,
        simple_model,
        small_dataset,
        training_args,
    ):
        """Test dataloader creation with Poisson sampling."""
        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        mock_poisson_loader = Mock()
        mock_dp_dataloader.from_data_loader.return_value = mock_poisson_loader

        mock_wrapped_loader = Mock()
        mock_wrap_data_loader.return_value = mock_wrapped_loader

        privacy_args = PrivacyArguments(noise_multiplier=1.0, poisson_sampling=True)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        dataloader = trainer.get_train_dataloader()

        # Should wrap with DPDataLoader for Poisson sampling
        mock_dp_dataloader.from_data_loader.assert_called_once()
        # Should wrap with batch memory manager
        mock_wrap_data_loader.assert_called_once()
        assert dataloader is mock_wrapped_loader

    @patch("dptrainer.trainer.prepare_module")
    @patch("dptrainer.trainer.wrap_data_loader")
    def test_get_train_dataloader_without_poisson(
        self,
        mock_wrap_data_loader,
        mock_wrap_model,
        simple_model,
        small_dataset,
        training_args,
    ):
        """Test dataloader creation without Poisson sampling."""
        mock_controller = Mock()
        mock_wrap_model.return_value = mock_controller

        mock_wrapped_loader = Mock()
        mock_wrap_data_loader.return_value = mock_wrapped_loader

        privacy_args = PrivacyArguments(noise_multiplier=1.0, poisson_sampling=False)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        dataloader = trainer.get_train_dataloader()

        # Should only wrap with batch memory manager, not DPDataLoader
        mock_wrap_data_loader.assert_called_once()
        assert dataloader is mock_wrapped_loader


class TestDPTrainerDetachModel:
    """Test model detachment."""

    @patch("dptrainer.trainer.prepare_module")
    def test_detach_model(
        self, mock_wrap_model, simple_model, small_dataset, training_args
    ):
        """Test that detach_model cleans up controller and returns model."""
        mock_controller = Mock()
        mock_controller.cleanup = Mock()
        mock_wrap_model.return_value = mock_controller

        privacy_args = PrivacyArguments(noise_multiplier=1.0)

        trainer = DPTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=small_dataset,
            privacy_args=privacy_args,
        )

        detached_model = trainer.detach_model()

        mock_controller.cleanup.assert_called_once()
        assert detached_model is simple_model


class TestDPTrainerPrivacyLogging:
    """Test privacy metrics logging behavior."""

    def test_privacy_log_mode_defaults_to_eval(self):
        assert PrivacyArguments().epsilon_log_mode == "eval"

    @patch("dptrainer.trainer.Trainer.log")
    def test_log_privacy_metrics_on_train_only(self, mock_super_log):
        trainer = DPTrainer.__new__(DPTrainer)
        trainer.model = Mock(training=True)
        trainer.dp_callback = Mock()
        trainer.dp_callback.get_privacy_metrics.return_value = {"privacy_epsilon": 1.23}
        trainer.privacy_args = PrivacyArguments(epsilon_log_mode="train")

        logs = {}

        DPTrainer.log(trainer, logs)

        assert logs == {"privacy_epsilon": 1.23}
        mock_super_log.assert_called_once_with(logs, None)

    @patch("dptrainer.trainer.Trainer.log")
    def test_log_privacy_metrics_on_eval_only(self, mock_super_log):
        trainer = DPTrainer.__new__(DPTrainer)
        trainer.model = Mock(training=False)
        trainer.dp_callback = Mock()
        trainer.dp_callback.get_privacy_metrics.return_value = {"privacy_epsilon": 1.23}
        trainer.privacy_args = PrivacyArguments(epsilon_log_mode="eval")

        logs = {}

        DPTrainer.log(trainer, logs)

        assert logs == {"eval_privacy_epsilon": 1.23}
        mock_super_log.assert_called_once_with(logs, None)

    @patch("dptrainer.trainer.Trainer.log")
    def test_log_privacy_metrics_uses_evaluation_metric_prefix(self, mock_super_log):
        trainer = DPTrainer.__new__(DPTrainer)
        trainer.model = Mock(training=False)
        trainer.dp_callback = Mock()
        trainer.dp_callback.get_privacy_metrics.return_value = {"privacy_epsilon": 1.23}
        trainer.privacy_args = PrivacyArguments(epsilon_log_mode="eval")

        validation_logs = {"validation_loss": 0.5}
        DPTrainer.log(trainer, validation_logs)

        dataset_logs = {"eval_test_loss": 0.5}
        DPTrainer.log(trainer, dataset_logs)

        runtime_logs = {"validation_runtime": 0.5}
        DPTrainer.log(trainer, runtime_logs)

        assert validation_logs == {"validation_loss": 0.5, "validation_privacy_epsilon": 1.23}
        assert dataset_logs == {"eval_test_loss": 0.5, "eval_test_privacy_epsilon": 1.23}
        assert runtime_logs == {"validation_runtime": 0.5, "validation_privacy_epsilon": 1.23}
        mock_super_log.assert_has_calls([
            call(validation_logs, None),
            call(dataset_logs, None),
            call(runtime_logs, None),
        ])

    @patch("dptrainer.trainer.Trainer.log")
    def test_log_privacy_metrics_on_both(self, mock_super_log):
        trainer = DPTrainer.__new__(DPTrainer)
        trainer.model = Mock(training=True)
        trainer.dp_callback = Mock()
        trainer.dp_callback.get_privacy_metrics.return_value = {"privacy_epsilon": 1.23}
        trainer.privacy_args = PrivacyArguments(epsilon_log_mode="both")

        train_logs = {"random_metric": 0.5}
        DPTrainer.log(trainer, train_logs)

        assert train_logs == {"random_metric": 0.5, "privacy_epsilon": 1.23}

        trainer.model.training = False
        eval_logs = {"random_metric": 0.5}
        DPTrainer.log(trainer, eval_logs)

        assert eval_logs == {"random_metric": 0.5, "eval_privacy_epsilon": 1.23}
        
        mock_super_log.assert_has_calls([call(train_logs, None), call(eval_logs, None)])
