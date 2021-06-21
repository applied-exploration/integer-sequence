from abc import ABC, abstractmethod
from typing import List, Tuple

class LearningAlgorithm(ABC):

    @abstractmethod
    def train(self, data: List[Tuple[List[int], str]]) -> None:
        pass

    @abstractmethod
    def infer(self, data: List[List[int]]) -> List[str]:
        pass