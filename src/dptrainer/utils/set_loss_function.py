def _unwrap_peft(model):
    """Unwrap PEFT wrappers if present."""
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.get_base_model()
    except ImportError:
        pass
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if hasattr(model, '_peft_config') or type(model).__name__.startswith('Peft'):
            return model.base_model.model
    return model


def set_loss_function(model, new_loss_function):
    """
    Set loss_function on the underlying task model only.

    Unwraps through known wrapper layers in order:
      1. GradSampleModule (_module)
      2. DDP / DataParallel (module) — via unwrap_model
      3. PEFT wrappers (base_model.model) — via _unwrap_peft

    Then sets loss_function on the final PreTrainedModel.
    """
    from transformers.modeling_utils import unwrap_model, PreTrainedModel

    # Unwrap Opacus GradSampleModule
    if hasattr(model, '_module'):
        model = model._module

    # Unwrap distributed wrappers
    model = unwrap_model(model)

    # Unwrap PEFT
    model = _unwrap_peft(model)

    # Final safety: only set on PreTrainedModel instances
    if isinstance(model, PreTrainedModel):
        model.loss_function = new_loss_function
    elif hasattr(model, 'loss_function'):
        model.loss_function = new_loss_function
    else:
        raise ValueError(
            f"Could not find a model with loss_function after unwrapping. "
            f"Final model type: {type(model).__name__}"
        )
