from dataclasses import dataclass
import numpy as np
from collections import deque, defaultdict
from typing import TypeVar
from .sampling import SamplingContext, SamplingFactory, SamplingTypes

ObsType = TypeVar("ObsType")

@dataclass(init=True)
class ExpSample:
    state:ObsType
    reward:float
    terminated:bool
    previous_state:ObsType
    action:int


class ExperienceBuffer:

    def __init__(self, size_of_buffer:int, sampling_method:str = 'SIMPLE_RANDOM') -> None:
        self.size_of_buffer = size_of_buffer
        self.sampling_method = SamplingTypes[sampling_method]
        self.buffer = deque(maxlen=self.size_of_buffer)


    def append(self, exp:ExpSample):
        self.buffer.append(exp)

    def get_samples(self, number_of_samples:int) -> dict:
        context = SamplingContext(self.sampling_method, 
                                samples=self.buffer, 
                                number_of_samples=number_of_samples)
        
        samples = SamplingFactory.get_samples(context)
        
        return self._create_data_dict(samples)
        
    def _create_data_dict(self, samples: list) -> dict:
        data_dict = defaultdict(list)
        
        for sample in samples:
            for name in ExpSample.__dataclass_fields__:
                data_dict[name].append(sample.__dict__[name])

        return dict(data_dict)
    
    def __len__(self):
        return len(self.buffer)