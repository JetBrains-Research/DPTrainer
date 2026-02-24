"""Shared test fixtures for dp-training tests."""

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from dptrainer import PrivacyArguments


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100, input_size=10, output_size=2):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randn(self.input_size),
            "labels": torch.randint(0, self.output_size, (1,)).item(),
        }


@pytest.fixture
def simple_model():
    """Fixture providing a simple model."""
    return SimpleModel()


@pytest.fixture
def simple_dataset():
    """Fixture providing a simple dataset."""
    return SimpleDataset(size=100)


@pytest.fixture
def small_dataset():
    """Fixture providing a small dataset for faster tests."""
    return SimpleDataset(size=20)


@pytest.fixture
def training_args(tmp_path):
    """Fixture providing basic TrainingArguments."""
    return TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=10,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )


@pytest.fixture
def privacy_args_with_noise():
    """Fixture providing PrivacyArguments with explicit noise multiplier."""
    return PrivacyArguments(
        accountant="rdp",
        noise_multiplier=1.0,
        per_sample_max_grad_norm=1.0,
        poisson_sampling=True,
    )


@pytest.fixture
def privacy_args_with_epsilon():
    """Fixture providing PrivacyArguments with target epsilon."""
    return PrivacyArguments(
        accountant="rdp",
        target_epsilon=3.0,
        per_sample_max_grad_norm=1.0,
        poisson_sampling=True,
    )


@pytest.fixture
def privacy_args_low():
    """Fixture providing low privacy arguments."""
    return PrivacyArguments.low_privacy()
