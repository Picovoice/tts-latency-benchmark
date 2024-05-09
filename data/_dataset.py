import random
from enum import Enum
from typing import (
    Any,
    Optional,
    Sequence,
)

from datasets import load_dataset


class TextDatasets(Enum):
    TASKMASTER2 = "taskmaster2"


class TextDataset:
    def __init__(self, seed: int = 777) -> None:
        self._sentences = []

        random.seed(seed)

    def get_random_sentences(self, num: int = 100) -> Sequence[str]:
        return random.sample(self.sentences, num)

    @property
    def sentences(self) -> Sequence[str]:
        return self._sentences

    def size(self) -> int:
        return len(self.sentences)

    @classmethod
    def create(cls, x: TextDatasets, **kwargs: Any) -> 'TextDataset':
        subclasses = {
            TextDatasets.TASKMASTER2: Taskmaster2Dataset,
        }

        if x not in subclasses:
            raise NotImplementedError(f"Cannot create {cls.__name__} of type `{x.value}`")

        return subclasses[x](**kwargs)


class Taskmaster2Dataset(TextDataset):
    NAME = "taskmaster2"
    CATEGORIES = ["flights", "food-ordering", "hotels", "movies", "music", "restaurant-search", "sports"]

    def __init__(self, categories: Optional[Sequence[str]] = None) -> None:
        super().__init__()

        self._categories = categories or self.CATEGORIES

        print("Loading dataset ...")
        sentences = set()
        for config in self._categories:
            dataset = load_dataset(self.NAME, config)['train']

            for entry in dataset:
                user_utterances = [ut["text"] for ut in entry["utterances"] if ut["speaker"].lower() == "user"]
                user_initial_query = user_utterances[0]
                sentences.add(user_initial_query)

        self._sentences = list(sentences)


__all__ = [
    "TextDatasets",
    "TextDataset",
]
