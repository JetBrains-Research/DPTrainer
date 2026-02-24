from transformers import EarlyStoppingCallback as HFEarlyStoppingCallback

class EarlyStoppingCallback(HFEarlyStoppingCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
