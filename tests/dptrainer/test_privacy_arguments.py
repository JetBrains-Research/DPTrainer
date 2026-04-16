"""Tests for PrivacyArguments dataclass."""

from dptrainer import PrivacyArguments


class TestPrivacyArgumentsPrecalculate:
    """Test the precalculate method."""

    def test_precalculate_with_noise_multiplier(self):
        """Test precalculate when noise_multiplier is already set."""
        args = PrivacyArguments(noise_multiplier=1.5)
        original_noise = args.noise_multiplier

        args.precalculate(num_samples=1000, sample_rate=0.01, steps=100)

        # Should not change when already set
        assert args.noise_multiplier == original_noise
        # target_delta should be set to 1/N
        assert args.target_delta == 1.0 / 1000

    def test_precalculate_target_delta_default(self):
        """Test that target_delta defaults to 1/N."""
        args = PrivacyArguments(noise_multiplier=1.0)
        args.precalculate(num_samples=5000, sample_rate=0.01, steps=100)
        assert args.target_delta == 1.0 / 5000

    def test_precalculate_target_delta_explicit(self):
        """Test that explicit target_delta is preserved."""
        args = PrivacyArguments(noise_multiplier=1.0, target_delta=1e-5)
        args.precalculate(num_samples=1000, sample_rate=0.01, steps=100)
        assert args.target_delta == 1e-5

    def test_precalculate_with_target_epsilon(self):
        """Test precalculate calculates noise_multiplier from target_epsilon."""
        args = PrivacyArguments(target_epsilon=3.0)
        assert args.noise_multiplier is None

        args.precalculate(num_samples=1000, sample_rate=0.01, steps=100)

        # Should calculate noise_multiplier
        assert args.noise_multiplier is not None
        assert args.noise_multiplier > 0
        assert args.target_delta == 1.0 / 1000


    def test_precalculate_defaults_to_zero_noise(self):
        """Test precalculate defaults to 0.0 noise when no target specified."""
        args = PrivacyArguments()
        assert args.noise_multiplier is None

        args.precalculate(num_samples=1000, sample_rate=0.01, steps=100)

        # Should default to 0.0 (no privacy)
        assert args.noise_multiplier == 0.0
