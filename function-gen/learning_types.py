from abc import ABC, abstractmethod
from typing import List, Tuple
from lang import Lang

class LearningAlgorithm(ABC):

    @abstractmethod
    def train(self, input_lang: Lang, output_lang: Lang, data: List[Tuple[List[int], str]]) -> None:
        pass

    @abstractmethod
    def infer(self, input_lang: Lang, output_lang: Lang, data: List[List[int]]) -> List[str]:
        pass