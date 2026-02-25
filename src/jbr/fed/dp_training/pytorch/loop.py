import os
import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Union, List, Optional

import torch
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import AbstractGradSampleModule
from opacus.utils.adaptive_clipping import PrivacyEngineAdaptiveClipping
from opacus.utils.batch_memory_manager import wrap_data_loader
from riskcal import CTDAccountant, get_beta_from_pld
from torch import multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn import Module
from torch.utils.data import Dataset, IterableDataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.optimization import get_scheduler

from jbr.fed.dp_training import PrivacyArguments
from jbr.fed.dp_training.hugging_face.patched import DataCollatorForCausalLM
from jbr.fed.dp_training.pytorch.training_args import TrainingArgumentsProtocol
from jbr.fed.dp_training.utils import set_loss_function_recursively

logger = logging.getLogger(__name__)

logging.getLogger("opacus.grad_sample.grad_sample_module_fast_gradient_clipping").setLevel(logging.WARNING)


def custom_training_loop(model: Module,
                         tokenizer: PreTrainedTokenizerBase,
                         train_dataset: Union[Dataset, IterableDataset],
                         eval_dataset: Union[Dataset, IterableDataset],
                         training_arguments: TrainingArgumentsProtocol,
                         privacy_arguments: PrivacyArguments = None,
                         model_input_names: List[str] = None,
                         early_stopping_patience: Optional[int] = None):
    """Run a custom PyTorch training loop with differential privacy support.

    Automatically detects available GPUs and uses distributed training when
    multiple GPUs are present. Supports checkpointing, early stopping, and
    privacy budget enforcement.

    Args:
        model (Module): The PyTorch model to train.
        tokenizer (PreTrainedTokenizerBase): The tokenizer for data collation.
        train_dataset (Union[Dataset, IterableDataset]): The training dataset.
        eval_dataset (Union[Dataset, IterableDataset]): The evaluation dataset.
        training_arguments (TrainingArgumentsProtocol): Training configuration arguments.
        privacy_arguments (PrivacyArguments): Privacy configuration. Defaults to low-privacy if not provided.
        model_input_names (List[str]): List of model input field names. Defaults to ["input_ids", "attention_mask", "labels"].
        early_stopping_patience (Optional[int]): Number of evaluations with no improvement before stopping.

    Returns:
        tuple: A tuple of (log_history, accountant) containing the training log history and the privacy accountant.
    """
    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:
        ctx = torch.multiprocessing.get_context('spawn')
        queue = ctx.SimpleQueue()
        proc_ctx = mp.spawn(_custom_training_loop_mp_wrapper,
                            args=(
                                gpu_count,
                                queue,
                                model,
                                tokenizer,
                                train_dataset,
                                eval_dataset,
                                training_arguments,
                                privacy_arguments,
                                model_input_names,
                                early_stopping_patience,
                            ),
                            nprocs=gpu_count,
                            join=False,
                            )

        try:
            proc_ctx.join()
            log_history, accountant = queue.get()

            return log_history, accountant
        except KeyboardInterrupt:
            for p in getattr(proc_ctx, "processes", []):
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            raise
        finally:
            queue.close()

    else:
        return _custom_training_loop_int(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_arguments=training_arguments,
            privacy_arguments=privacy_arguments,
            model_input_names=model_input_names,
            early_stopping_patience=early_stopping_patience,
        )


def _custom_training_loop_mp_wrapper(rank: int, world_size: int, queue: mp.Queue, *args, **kwargs):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '35555'
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        if rank != 0:
            logger.setLevel(logging.ERROR)

        torch.cuda.set_device(rank)

        torch.distributed.init_process_group(
            rank=rank, world_size=world_size, init_method="env://"
        )

        log_history, accountant = _custom_training_loop_int(*args, rank=rank, world_size=world_size, **kwargs)

        if rank == 0:
            queue.put((log_history, accountant))
    finally:
        destroy_process_group()


def _custom_training_loop_int(model: Module,
                              tokenizer: PreTrainedTokenizerBase,
                              train_dataset: Union[Dataset, IterableDataset],
                              eval_dataset: Union[Dataset, IterableDataset],
                              training_arguments: TrainingArgumentsProtocol,
                              privacy_arguments: PrivacyArguments = None,
                              model_input_names: List[str] = None,
                              early_stopping_patience: Optional[int] = None,
                              rank: int = 0, world_size: int = 1, ):
    """
    Custom training loop for the model.
    """
    main_process = (rank == 0)

    if torch.cuda.is_available():
        model.to(rank)

    model_device = model.device
    model.train()

    if not model_input_names:
        model_input_names = ["input_ids", "attention_mask", "labels"]

    # Filter datasets to only include model inputs
    def filter_model_inputs(batch_samples):
        return {k: v for k, v in batch_samples.items() if k in model_input_names}

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    def custom_collate_fn(batch_samples):
        # Filter each item in the batch to only include model inputs
        filtered_batch = [filter_model_inputs(item) for item in batch_samples]
        return data_collator(filtered_batch)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps,
        shuffle=(not isinstance(train_dataset, IterableDataset)) and not (
                privacy_arguments and privacy_arguments.poisson_sampling),
        collate_fn=custom_collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_arguments.per_device_eval_batch_size,
        collate_fn=custom_collate_fn,
    )

    num_steps_per_epoch = len(train_dataloader)
    num_epochs = int(training_arguments.num_train_epochs)
    num_steps = int(num_steps_per_epoch * num_epochs)
    num_warmup_steps = int(num_steps * training_arguments.warmup_ratio)

    unwrapped_model = model
    if not hasattr(unwrapped_model.loss_function, "reduction"):
        setattr(unwrapped_model.loss_function, "reduction", "mean")

    if world_size != 1:
        model = DPDDP(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_arguments.learning_rate,
        betas=(training_arguments.adam_beta1, training_arguments.adam_beta2),
        eps=training_arguments.adam_epsilon,
        weight_decay=training_arguments.weight_decay,
    )

    scheduler = get_scheduler(
        name=training_arguments.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps,
        scheduler_specific_kwargs=training_arguments.lr_scheduler_kwargs
    )

    # Privatize model, optimizer, dataloader
    if privacy_arguments is None:
        warnings.warn("This pipeline requires privacy configuration to be set. Using low-privacy equivalent.")
        privacy_arguments = PrivacyArguments.low_privacy()

    dataset_size = len(train_dataset)
    sample_rate = (training_arguments.per_device_train_batch_size
                   * training_arguments.gradient_accumulation_steps / dataset_size)

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        privacy_engine = (PrivacyEngineAdaptiveClipping(accountant=privacy_arguments.accountant)
                          if privacy_arguments.clipping == "adaptive" and "ghost" in privacy_arguments.grad_sample_mode
                          else PrivacyEngine(accountant=privacy_arguments.accountant))

    adaptive_kwargs = {
        'max_clipbound': privacy_arguments.max_clipbound,
        'min_clipbound': privacy_arguments.min_clipbound,
        'clipbound_learning_rate': privacy_arguments.clipbound_learning_rate,
        'target_unclipped_quantile': privacy_arguments.target_unclipped_quantile,
        'unclipped_num_std': privacy_arguments.unclipped_num_std} \
        if privacy_arguments.clipping == "adaptive" else {}
    private_components = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=unwrapped_model.loss_function,
        data_loader=train_dataloader,
        noise_multiplier=privacy_arguments.noise_multiplier,
        max_grad_norm=privacy_arguments.per_sample_max_grad_norm,
        clipping=(privacy_arguments.clipping
                  if not (
                privacy_arguments.clipping == "adaptive" and "ghost" in privacy_arguments.grad_sample_mode)
                  else "flat"),  # it is adaptive but via loss
        grad_sample_mode=privacy_arguments.grad_sample_mode,
        poisson_sampling=privacy_arguments.poisson_sampling,
        **adaptive_kwargs
    )
    if "ghost" in privacy_arguments.grad_sample_mode:
        model, optimizer, criterion, train_dataloader = private_components
        set_loss_function_recursively(model, criterion)
    else:
        model, optimizer, train_dataloader = private_components

    train_dataloader = wrap_data_loader(
        data_loader=train_dataloader,
        optimizer=optimizer,
        max_batch_size=training_arguments.per_device_train_batch_size
    )

    # Checkpoint and state initialization
    global_step = 0
    log_history = []
    best_eval_loss = float("inf")
    best_global_step = None
    stop = False
    start_epoch = 0

    tensorboard_logging_dir = training_arguments.output_dir + "/tensorboard"
    os.makedirs(tensorboard_logging_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_logging_dir)

    checkpoint_exists = (
            training_arguments.output_dir is not None
            and bool(list(Path(training_arguments.output_dir).glob("checkpoint-*")))
    )

    if checkpoint_exists:
        checkpoint_dir = _find_latest_checkpoint(training_arguments.output_dir)
        if checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
            checkpoint_state = _load_checkpoint(model, checkpoint_dir,
                                                privacy_engine=privacy_engine,
                                                optimizer=optimizer,
                                                scheduler=scheduler
                                                )

            # Restore training state
            global_step = checkpoint_state.get('global_step', 0)
            best_global_step = checkpoint_state.get('best_global_step', global_step)
            start_epoch = checkpoint_state.get('epoch', 1)
            best_eval_loss = checkpoint_state.get('best_eval_loss', float("inf"))
            log_history = checkpoint_state.get('log_history', [])

            if early_stopping_patience is not None and best_global_step is not None:
                patience = (global_step - best_global_step) // training_arguments.eval_steps
                if patience >= early_stopping_patience:
                    stop = True

            if privacy_engine.accountant.history:
                if isinstance(privacy_engine.accountant, CTDAccountant):
                    pld = privacy_engine.accountant.get_pld(grid_step=0.01)
                    privacy_epsilon = pld.get_epsilon_for_delta(delta=privacy_arguments.target_delta)
                    privacy_beta = get_beta_from_pld(pld, privacy_arguments.target_alpha)
                    if privacy_arguments.target_beta is not None and privacy_beta <= privacy_arguments.target_beta:
                        stop = True
                else:
                    with warnings.catch_warnings(category=UserWarning, action="ignore"):
                        privacy_epsilon = privacy_engine.accountant.get_epsilon(privacy_arguments.target_delta)

                if (
                        privacy_arguments.target_epsilon is not None and privacy_epsilon >= privacy_arguments.target_epsilon):
                    stop = True

    # If resuming, calculate which step within the current epoch to start from
    start_step_in_epoch = 0
    if checkpoint_exists and global_step > 0:
        completed_epochs = global_step // num_steps_per_epoch
        start_step_in_epoch = global_step % num_steps_per_epoch
        start_epoch = completed_epochs
        logger.info(f"Resuming from {global_step} global step, "
                    f"epoch {start_epoch}, step {start_step_in_epoch} within epoch")

    # loop
    for epoch in range(start_epoch, num_epochs):
        if stop:
            break

        # Skip batches if resuming from the middle of the epoch
        dataloader_iter = iter(train_dataloader)
        if epoch == start_epoch and start_step_in_epoch > 0:
            # Skip the already processed batches
            for _ in range(start_step_in_epoch):
                try:
                    next(dataloader_iter)
                except StopIteration:
                    break

        with tqdm(desc=f"Epoch {epoch + 1}/{int(training_arguments.num_train_epochs)}",
                  initial=start_step_in_epoch if epoch == start_epoch else 0,
                  total=num_steps_per_epoch, disable=not main_process) as tqdm_steps:
            loss_accumulator = 0.0
            for batch in dataloader_iter:
                if stop:
                    break

                batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss_accumulator += loss.item() / training_arguments.gradient_accumulation_steps
                tqdm_steps.set_postfix_str(f"loss = {loss.item():.3f}")

                loss.backward()
                optimizer.step()

                if optimizer._is_last_step_skipped:
                    continue  # until the actual batch is processed completely

                global_step += 1
                tqdm_steps.update(1)

                scheduler.step()
                privacy_engine.accountant.step(noise_multiplier=optimizer.noise_multiplier, sample_rate=sample_rate)

                # Stop if reached the configured total number of steps
                if global_step >= num_steps:
                    stop = True

                # Logging
                if global_step % training_arguments.logging_steps == 0 or global_step == num_steps:
                    grad_norms = [p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=1) for p in model.parameters()
                                  if getattr(p, "grad_sample", None) is not None]
                    per_sample_grad_norm = torch.stack(grad_norms, dim=0).norm(2, dim=0) if grad_norms else torch.zeros(
                        0)
                    avg_grad_norm = torch.mean(per_sample_grad_norm).item()
                    history_item = {"epoch": global_step / num_steps_per_epoch,
                                    "learning_rate": scheduler.get_last_lr()[0],
                                    "loss": loss_accumulator,
                                    "step": global_step,
                                    "grad_norm": avg_grad_norm,
                                    }
                    if privacy_engine.accountant.history:
                        if isinstance(privacy_engine.accountant, CTDAccountant):
                            pld = privacy_engine.accountant.get_pld(grid_step=0.01)
                            history_item["eval_privacy_epsilon"] = pld.get_epsilon_for_delta(
                                delta=privacy_arguments.target_delta)
                            history_item["eval_privacy_beta"] = get_beta_from_pld(pld, privacy_arguments.target_alpha)
                            if (privacy_arguments.target_beta is not None
                                    and history_item["eval_privacy_beta"] <= privacy_arguments.target_beta):
                                logger.info(
                                    "Privacy budget reached! "
                                    f"beta@{privacy_arguments.target_alpha:.5f}={history_item['eval_privacy_beta']:.3f}")
                                stop = True

                        else:
                            with warnings.catch_warnings(category=UserWarning, action="ignore"):
                                history_item["eval_privacy_epsilon"] = privacy_engine.accountant.get_epsilon(
                                    privacy_arguments.target_delta)

                        if (privacy_arguments.target_epsilon is not None and
                                history_item["eval_privacy_epsilon"] >= privacy_arguments.target_epsilon):
                            logger.info(
                                "Privacy budget reached! "
                                f"epsilon@{privacy_arguments.target_delta:.5f}={history_item['eval_privacy_epsilon']:.3f}")
                            stop = True

                    log_history.append(history_item)
                    if main_process:
                        tqdm_steps.write(str(history_item))
                        writer.add_scalar("train/loss", loss_accumulator, global_step)
                        writer.add_scalar("train/learning_rate", history_item["learning_rate"], global_step)
                        writer.add_scalar("train/grad_norm", history_item["grad_norm"], global_step)

                        if "eval_privacy_epsilon" in history_item:
                            writer.add_scalar("privacy/epsilon", history_item["eval_privacy_epsilon"], global_step)
                        if "eval_privacy_beta" in history_item:
                            writer.add_scalar("privacy/beta", history_item["eval_privacy_beta"], global_step)
                    tqdm_steps.set_postfix_str(f"loss = {loss_accumulator:.3f}")
                loss_accumulator = 0.0

                # evaluate
                should_evaluate = global_step % training_arguments.eval_steps == 0 or global_step == num_steps
                if should_evaluate:
                    eval_loss = _evaluate_model(model, eval_dataloader, rank)

                    eval_history_item = {
                        "epoch": global_step / num_steps_per_epoch,
                        "eval_loss": eval_loss,
                        "step": global_step
                    }
                    log_history.append(eval_history_item)
                    if main_process:
                        writer.add_scalar("eval/loss", eval_loss, global_step)
                        tqdm_steps.write(str(eval_history_item))

                    # update early stopping (if early stoping then avoid the last)
                    if eval_loss < best_eval_loss and not stop:
                        best_eval_loss = eval_loss
                        best_global_step = global_step

                    # early stopping check
                    if early_stopping_patience is not None and best_global_step is not None:
                        patience = (global_step - best_global_step) // training_arguments.eval_steps
                        if patience >= early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered! No improvement for {patience} evaluations.")
                            stop = True

                # Save checkpoint
                should_save = main_process and (global_step % training_arguments.save_steps == 0
                                                or global_step == num_steps
                                                or best_global_step == global_step
                                                or stop)
                if should_save:
                    _save_checkpoint(
                        model,
                        training_arguments.output_dir,
                        global_step,
                        epoch,
                        optimizer,
                        scheduler,
                        best_eval_loss,
                        best_global_step,
                        log_history,
                        training_arguments,
                        privacy_engine,
                        num_steps,
                    )
                    _remove_stale_checkpoints(training_arguments.output_dir, best_global_step=best_global_step)

    # load the last best model if early stopping
    if best_global_step is not None and best_global_step != global_step:
        best_checkpoint_path = os.path.join(training_arguments.output_dir, f"checkpoint-{best_global_step}")
        _load_checkpoint(model, best_checkpoint_path, privacy_engine)

    try:
        writer.flush()
        writer.close()
    except Exception as e:
        logger.error(f"Error while closing TensorBoard writer: {e}")

    return log_history, privacy_engine.accountant


def _evaluate_model(model: Module, eval_dataloader: DataLoader, rank: int) -> float:
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, leave=False, desc="Evaluating", disable=rank != 0):
            batch = {k: v.to(rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            batch_loss = outputs.loss.item()
            batch_size = batch["input_ids"].size(0)
            total_loss += batch_loss * batch_size
            total_samples += batch_size

    model.train()
    avg_loss = total_loss / total_samples
    logger.debug(f"Average eval loss: {avg_loss:.6f} over {total_samples} samples")
    return avg_loss


def _save_checkpoint(model: Union[Module, AbstractGradSampleModule],
                     output_dir: str, global_step: int, epoch: int,
                     optimizer, scheduler,
                     best_eval_loss: float, best_global_step: int,
                     log_history: list, training_arguments: TrainingArgumentsProtocol,
                     privacy_engine: PrivacyEngine,
                     num_steps: int):
    """Save training checkpoint in HuggingFace Trainer format."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    state_checkpoint_path = os.path.join(checkpoint_dir, "state.cpt")
    checkpoint_dict = {
        "scheduler_state_dict": scheduler.state_dict()
    }
    privacy_engine.save_checkpoint(
        path=state_checkpoint_path,
        module=model,
        optimizer=optimizer,
        checkpoint_dict=checkpoint_dict
    )

    trainer_state = {
        "global_step": global_step,
        "epoch": epoch,
        "num_steps": num_steps,
        "num_train_epochs": training_arguments.num_train_epochs,
        "log_history": log_history,
        "best_global_step": best_global_step,
        "best_eval_loss": best_eval_loss if best_eval_loss != float("inf") else None,
    }

    with open(os.path.join(checkpoint_dir, "trainer_state.json"), 'w') as f:
        json.dump(trainer_state, f, indent=2)

    return checkpoint_dir


def _load_checkpoint(model: Union[Module, AbstractGradSampleModule], checkpoint_dir: str,
                     privacy_engine: PrivacyEngine,
                     optimizer=None, scheduler=None):
    """Load training checkpoint in HuggingFace Trainer format."""
    state_checkpoint_path = os.path.join(checkpoint_dir, "state.cpt")
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        state_dict = privacy_engine.load_checkpoint(
            path=state_checkpoint_path,
            module=model,
            optimizer=optimizer,
        )
    if scheduler:
        scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    model.train()

    # Load trainer state
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    return trainer_state


def _find_latest_checkpoint(output_dir: str):
    """Find the latest checkpoint in the output directory."""
    if not output_dir or not os.path.exists(output_dir):
        return None

    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            checkpoint_path = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_path):
                try:
                    step = int(item.split("-")[1])
                    checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue

    if checkpoint_dirs:
        # Return the checkpoint with the highest step number
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs[-1][1]

    return None


def _remove_stale_checkpoints(output_dir: str, best_global_step: int = None, keep_last: bool = True):
    """Removes stale checkpoints in the output directory."""
    if not output_dir or not os.path.exists(output_dir):
        return

    checkpoint_dirs = {}
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            checkpoint_path = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_path):
                try:
                    step = int(item.split("-")[1])
                    checkpoint_dirs[step] = checkpoint_path
                except ValueError:
                    continue

    if checkpoint_dirs and keep_last:
        latest_checkpoint = max(checkpoint_dirs.keys())
        checkpoint_dirs.pop(latest_checkpoint, None)

    if checkpoint_dirs and best_global_step:
        checkpoint_dirs.pop(best_global_step, None)

    for _, dir in checkpoint_dirs.items():
        shutil.rmtree(dir)

    return
