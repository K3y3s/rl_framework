from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
from enum import Enum, auto

from typing import TypeVar

ExpSample = TypeVar("ExpSample")


class SamplingTypes(Enum):
    SIMPLE_RANDOM = auto()


class SamplingContext:
    def __init__(
            self, 
            sampling_method:str, 
            samples:Sequence, 
            number_of_samples:int
            ) -> None:
        
        self.sampling_method = sampling_method
        self.samples = samples
        self.number_of_samples = number_of_samples

class Sampling(ABC):

    @abstractmethod
    def get_samples(self) -> np.array:
        """crate samples"""
        raise  NotImplementedError


class SimpleRandomSampling(Sampling):

    def get_samples(
        samples:list[ExpSample], 
        number_of_samples:int
        ) -> list[ExpSample]:
        return [samples[idx] for idx in np.random.choice(range(len(samples)), number_of_samples, replace = False)]
        

class SamplingFactory:
    @staticmethod
    def get_samples(context:SamplingContext) -> np.array:
        if context.sampling_method == SamplingTypes.SIMPLE_RANDOM:
            return SimpleRandomSampling.get_samples(
                context.samples, 
                context.number_of_samples
                )
