"""Tests for plot functions in dp_training package."""

import os
import tempfile

import pandas as pd

from jbr.fed.dp_training.plots import plot_losses, plot_privacy_epsilon, plot_privacy_beta


class TestPlotLosses:
    """Test plot_losses function."""

    def test_plot_losses_creates_files(self):
        """Test that plot_losses creates expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'loss': 1.5},
                {'step': 200, 'loss': 1.2},
                {'step': 200, 'eval_loss': 1.3},
                {'step': 400, 'loss': 1.0},
                {'step': 400, 'eval_loss': 1.1},
            ]

            plot_losses(tmpdir, log_history)

            # Check that PNG file was created
            assert os.path.exists(os.path.join(tmpdir, "loss_history.png"))

            # Check that CSV file was created
            csv_path = os.path.join(tmpdir, "loss_history.csv")
            assert os.path.exists(csv_path)

            # Verify CSV contents
            df = pd.read_csv(csv_path)
            assert 'step' in df.columns
            assert 'train_loss' in df.columns
            assert 'eval_loss' in df.columns
            assert len(df) >= 2

    def test_plot_losses_with_empty_history(self):
        """Test plot_losses with empty log history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = []

            plot_losses(tmpdir, log_history)

            # Should still create files even with empty data
            assert os.path.exists(os.path.join(tmpdir, "loss_history.png"))
            assert os.path.exists(os.path.join(tmpdir, "loss_history.csv"))

    def test_plot_losses_with_train_only(self):
        """Test plot_losses with only training losses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'loss': 1.5},
                {'step': 200, 'loss': 1.2},
                {'step': 300, 'loss': 1.0},
            ]

            plot_losses(tmpdir, log_history)

            assert os.path.exists(os.path.join(tmpdir, "loss_history.png"))

            csv_path = os.path.join(tmpdir, "loss_history.csv")
            df = pd.read_csv(csv_path)
            assert len(df[df['train_loss'].notna()]) == 3


class TestPlotPrivacyEpsilon:
    """Test plot_privacy_epsilon function."""

    def test_plot_privacy_epsilon_creates_file(self):
        """Test that plot_privacy_epsilon creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'eval_privacy_epsilon': 1.0},
                {'step': 200, 'eval_privacy_epsilon': 2.5},
                {'step': 300, 'eval_privacy_epsilon': 4.0},
            ]
            delta = 1e-5

            plot_privacy_epsilon(tmpdir, log_history, delta)

            assert os.path.exists(os.path.join(tmpdir, "eval_privacy_epsilon.png"))

    def test_plot_privacy_epsilon_with_empty_history(self):
        """Test plot_privacy_epsilon with empty log history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = []
            delta = 1e-5

            plot_privacy_epsilon(tmpdir, log_history, delta)

            # Should still create file with zero epsilon at step 0
            assert os.path.exists(os.path.join(tmpdir, "eval_privacy_epsilon.png"))

    def test_plot_privacy_epsilon_different_deltas(self):
        """Test plot_privacy_epsilon with different delta values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'eval_privacy_epsilon': 1.0},
                {'step': 200, 'eval_privacy_epsilon': 2.0},
            ]

            for delta in [1e-3, 1e-5, 1e-7]:
                plot_privacy_epsilon(tmpdir, log_history, delta)
                assert os.path.exists(os.path.join(tmpdir, "eval_privacy_epsilon.png"))


class TestPlotPrivacyBeta:
    """Test plot_privacy_beta function."""

    def test_plot_privacy_beta_creates_file(self):
        """Test that plot_privacy_beta creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'eval_privacy_beta': 0.5},
                {'step': 200, 'eval_privacy_beta': 0.3},
                {'step': 300, 'eval_privacy_beta': 0.2},
            ]
            alpha = 0.05

            plot_privacy_beta(tmpdir, log_history, alpha)

            assert os.path.exists(os.path.join(tmpdir, "eval_privacy_beta.png"))

    def test_plot_privacy_beta_with_empty_history(self):
        """Test plot_privacy_beta with empty log history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = []
            alpha = 0.05

            plot_privacy_beta(tmpdir, log_history, alpha)

            # Should still create file with zero beta at step 0
            assert os.path.exists(os.path.join(tmpdir, "eval_privacy_beta.png"))

    def test_plot_privacy_beta_different_alphas(self):
        """Test plot_privacy_beta with different alpha values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_history = [
                {'step': 100, 'eval_privacy_beta': 0.5},
                {'step': 200, 'eval_privacy_beta': 0.3},
            ]

            for alpha in [0.01, 0.05, 0.1]:
                plot_privacy_beta(tmpdir, log_history, alpha)
                assert os.path.exists(os.path.join(tmpdir, "eval_privacy_beta.png"))
