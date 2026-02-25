from typing import Union

from datasets import Dataset, IterableDataset


def convert_to_static(dataset: Union[IterableDataset, Dataset], num_samples: int = None) -> Dataset:
    """Convert an iterable or static dataset to a static `Dataset`.

    Args:
        dataset (Union[IterableDataset, Dataset]): The dataset to convert.
        num_samples (int): If provided, only take this many samples from the dataset.

    Returns:
        Dataset: A static HuggingFace Dataset.
    """
    if num_samples is not None:
        dataset = dataset.take(num_samples)

    return Dataset.from_list(list(dataset))