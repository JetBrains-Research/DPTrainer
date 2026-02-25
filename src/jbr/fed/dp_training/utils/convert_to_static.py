from typing import Union

from datasets import Dataset, IterableDataset


def convert_to_static(dataset: Union[IterableDataset, Dataset], num_samples: int = None) -> Dataset:
    if num_samples is not None:
        dataset = dataset.take(num_samples)

    return Dataset.from_list(list(dataset))