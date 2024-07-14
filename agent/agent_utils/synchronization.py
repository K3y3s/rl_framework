from abc import ABC, abstractmethod
from torch.nn import Module
from dataclasses import dataclass
from typing import Callable

class Synchronization(ABC):

    @abstractmethod
    def make_synchronization(self, current_step:int):
        pass

class NoSynchronization(Synchronization):

    def make_synchronization(self, current_step:int):
        pass

class SimpleSynchronization(Synchronization):

    def __init__(self, nn:Module, nn_to_synchronize:Module, synchronization:int) -> None:
        self.nn = nn
        self.nn_to_synchronize
        self.synchronization = synchronization

    def make_synchronization(self, current_step:int) -> None:
        if current_step % self.synchronization == 0:
            pass

@dataclass
class SyncContex:
    nn:Module
    synchronization:int
    nn_to_synchornize:Module | None = None

class SynchronizationBuilder:

    def __init__(self, sync_context: SyncContex) -> None:
        self.sync = sync_context

    def build_synchronization(self) -> callable:
        
        if self.sync.synchronization == 0:
            return NoSynchronization().make_synchronization

        else:
            return SimpleSynchronization(
                nn = self.sync.nn, 
                nn_to_synchronize=self.sync.nn_to_synchronize,
                synchronization=self.sync.synchronization
                ).make_synchronization

