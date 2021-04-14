"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch

SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(self, data:SequenceOrTensor, targets:SequenceOrTensor, transform: Callable = None,
                 target_transform: Callable = None) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must have the same length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Return the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a data sample x and its corresponding target y
        :param index: the inde
        :return:
        (x,y)
        """
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    <S> is the special token to indicate beginning of a string sequence.
    <E> is the special token to indicate end of a string sequence.
    <P> is special token to pad sequences.

    Args:
        strings: The sequence of strings to map.
        mapping: A token to index mapping.
        length: The max length of the string sequences.

    Returns:

    """

    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for j, token in enumerate(tokens):
            labels[i, j] = mapping[token]

    return labels


def split_dataset(dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base dataset into 2 base datasets of size fraction * size and (1-fraction) * size,
    where size is the length of the base dataset.

    """
    split_a_size = int(len(dataset) * fraction)
    split_b_size = len(dataset) - split_a_size

    return torch.utils.data.random_split(dataset, [split_a_size, split_b_size],
                                         generator=torch.Generator().manual_seed(seed))


