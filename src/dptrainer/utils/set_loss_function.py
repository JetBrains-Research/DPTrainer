def set_loss_function_recursively(model, new_loss_function, max_depth=10):
    """
    Recursively set the loss_function property on all models in the wrapper hierarchy.

    This function handles various model wrappers like:
    - GradSampleModule (has _module attribute)
    - PeftModelForCausalLM (has .base_model attribute)
    - LoraModel (has .model attribute)
    - DistributedDataParallel (has .module attribute)

    Args:
        model: The wrapped model to process
        new_loss_function: The new loss function to set
        max_depth (int): Maximum recursion depth to prevent infinite loops

    Raises:
        ValueError: If maximum depth is reached or circular reference is detected
    """
    visited = set()

    def _set_recursive(current_model, depth=0):

        if depth >= max_depth:
            raise ValueError(f"Maximum unwrapping depth {max_depth} reached")

        model_id = id(current_model)
        if model_id in visited:
            return
        visited.add(model_id)

        # Set loss_function if it exists on the current model
        if hasattr(current_model, 'loss_function'):
            current_model.loss_function = new_loss_function

        # Continue recursively through wrapper attributes
        wrapper_attrs = [
            '_module',  # GradSampleModule
            'base_model',  # PeftModelForCausalLM, PEFT models
            'model',  # LoraModel
            'module',  # DistributedDataParallel, DataParallel
        ]

        for attr_name in wrapper_attrs:
            if hasattr(current_model, attr_name):
                wrapped_model = getattr(current_model, attr_name)
                # Make sure it's actually a model object and not None
                if wrapped_model is not None and hasattr(wrapped_model, '__class__'):
                    try:
                        _set_recursive(wrapped_model, depth + 1)
                    except (ValueError, AttributeError):
                        continue

    _set_recursive(model)
    return
