from abc import ABC, abstractmethod
from torch.nn import Module
from dataclasses import dataclass
from copy import deepcopy

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
        self.nn_to_synchronize = nn_to_synchronize
        self.synchronization = synchronization

    def make_synchronization(self, current_step:int) -> None:
        if current_step % self.synchronization == 0:
            for target, online in zip (self.nn_to_synchronize.parameters(), 
                                       self.nn.parameters()):
                target.data.copy_(online.data)
            #print(f'Synchronization done after {current_step}')

@dataclass
class SyncContext:
    nn:Module
    synchronization:int
    nn_to_synchronize:Module | None = None

class SynchronizationBuilder:

    def __init__(self, sync_context: SyncContext) -> None:
        self.sync = sync_context

    def build_synchronization(self) -> Synchronization:
        
        if self.sync.synchronization == 0:
            return NoSynchronization().make_synchronization

        else:
            return SimpleSynchronization(
                nn = self.sync.nn, 
                nn_to_synchronize=self.sync.nn_to_synchronize,
                synchronization=self.sync.synchronization
                ).make_synchronization

