import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_losses(output_dir: str, log_history):
    """Plot training and evaluation loss history and save as PNG and CSV.

    Args:
        output_dir (str): Directory to save the plot and CSV file.
        log_history: List of log dictionaries from the trainer.
    """
    train_logs = [log for log in log_history if 'loss' in log and 'step' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log and 'step' in log]

    train_steps = [log['step'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]
    eval_steps = [log['step'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]

    plt.figure()
    plt.plot(train_steps, train_losses, label="Training")
    plt.plot(eval_steps, eval_losses, label="Evaluation")
    plt.title("Training and Evaluation Loss History")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_history.png"))
    plt.close()

    train_df = pd.DataFrame({'step': train_steps, 'train_loss': train_losses})
    eval_df = pd.DataFrame({'step': eval_steps, 'eval_loss': eval_losses})
    loss_df = pd.merge(train_df, eval_df, on='step', how='outer').sort_values('step')
    loss_df.to_csv(os.path.join(output_dir, f"loss_history.csv"), index=False)


def plot_privacy_epsilon(output_dir: str, log_history, delta):
    """Plot privacy epsilon history over training steps and save as PNG.

    Args:
        output_dir (str): Directory to save the plot.
        log_history: List of log dictionaries from the trainer.
        delta: The target delta value used in the plot legend.
    """
    train_logs = [log for log in log_history if 'eval_privacy_epsilon' in log and 'step' in log]

    train_steps = [0] + [log['step'] for log in train_logs]
    train_epsilon = [0] + [log['eval_privacy_epsilon'] for log in train_logs]

    plt.figure()
    plt.plot(train_steps, train_epsilon, label=f"δ={delta:.3e}")
    plt.title("Privacy (ε, δ) History")
    plt.xlabel("Steps")
    plt.xlim(left=0)
    plt.ylabel("ε")
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eval_privacy_epsilon.png"))
    plt.close()


def plot_privacy_beta(output_dir: str, log_history, alpha):
    """Plot privacy beta (trade-off function) history over training steps and save as PNG.

    Args:
        output_dir (str): Directory to save the plot.
        log_history: List of log dictionaries from the trainer.
        alpha: The target alpha (false positive rate) value used in the plot legend.
    """
    train_logs = [log for log in log_history if 'eval_privacy_beta' in log and 'step' in log]

    train_steps = [0] + [log['step'] for log in train_logs]
    train_beta = [0] + [log['eval_privacy_beta'] for log in train_logs]

    plt.figure()
    plt.plot(train_steps, train_beta, label=f"α={alpha:.3e}")
    plt.title("Privacy Trade-off History")
    plt.xlabel("Steps")
    plt.xlim(left=0)
    plt.ylabel("β")
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"eval_privacy_beta.png"))
    plt.close()
