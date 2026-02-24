from transformers import EarlyStoppingCallback as HFEarlyStoppingCallback


class EarlyStoppingCallback(HFEarlyStoppingCallback):
    """Checkpoint-aware early stopping callback.

    Extends HuggingFace's `EarlyStoppingCallback` to also check the patience counter
    at the beginning of training, so that training stops immediately when resuming
    from a checkpoint that already exceeded the patience.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        """Check if patience is already exceeded when resuming from checkpoint."""
        super().on_train_begin(args, state, control, **kwargs)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
