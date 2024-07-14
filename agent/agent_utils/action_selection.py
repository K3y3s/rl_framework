from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from torch.nn import Module
from torch import Tensor, argmax, softmax
from enum import Enum, auto


class ActionMethods(Enum):
    RANDOM = auto()
    NEURAL_NETWORK = auto()

class PolicyMethods(Enum):
    GREEDY = auto()
    SOFT_MAX = auto()
    

@dataclass
class ActionContext:
    actions_number:int | None # number of actions to select
    action_methods:str | None = None # do random function
    epsilon:float | None = None # randomization factor for exploration
    nn: Module | None = None # neural network used for actions
    policy:str | None = None

    def __post_init__(self):
        self.action_methods =ActionMethods[self.action_methods]
        self.policy = PolicyMethods[self.policy]
        
############################################ Policy

class Policy(ABC):

    @abstractmethod
    def choose_action(self, action_state_values):
        raise NotImplementedError

class Greedy(Policy):
    
    def choose_action(self, action_state_values:Tensor):
        return argmax(action_state_values)
    
class SoftMax(Policy):

    def choose_action(self, action_state_values):
        #softmax method
        pass
                
############################################ Exploration

class Exploration(ABC):

    @abstractmethod
    def make_exploration(self) -> int:
        raise NotImplementedError

class SimpleExploration(Exploration):

    def make_exploration(actions:int) -> int:
        return np.random.choice(range(actions))


############################################## Action Selection

class ActionSelection(ABC):

    def __init__(self, action_context:ActionContext) -> None:
        self.action_cont = action_context

    @abstractmethod
    def compute_actions_states(self, state) -> int:
        raise NotImplementedError


class RandomActionSelection(ActionSelection):
    def __init__(self, action_context:ActionContext) -> None:
        super().__init__(action_context)

    def compute_actions_states(self, state) -> int:
        return np.random.randint(0, self.action_cont.actions_number)


class NeuralNetworkAction(ActionSelection):
    def __init__(self, action_context:ActionContext) -> None:
            super().__init__(action_context)

    def compute_actions_states(self, state) -> int:
        actions =  self.action_cont.nn(state)
        return actions

############################################ Actions

class Action:
    """
        How action is determined:
        1) Do exploration or not

        2) Compute action - state values Q(a, s) - potentail value and reward commint 
        from taking action a under state s
 
        3) Choose action according to the policy

    """

    def __init__(self, action_context:ActionContext):
        self.action_context = action_context

    def make_exploration() -> int:
        raise NotImplementedError

    def compute_actions_states() -> int | list[float]:
        raise NotImplementedError

    def choose_action(self, action_state_values: list[float]) -> int:
        raise NotImplementedError

    def compute_action(self, states = None) -> int | float:
        
        if np.random.random() < self.action_context.epsilon:
            action_number = self.make_exploration(self.action_context.actions_number)
        
        else:
            actions_vals = self.compute_actions_states(states)

            if isinstance(actions_vals, list):
                action_number = self.choose_action(actions_vals)
            else:
                action_number = actions_vals

        return action_number


class ActionBuilder:
    def __init__(self, action_context:ActionContext):
        self.a = Action(action_context)
        self.action_context = action_context

    def _build_policy(self) -> None:
        if self.action_context == PolicyMethods.GREEDY:
            self.a.choose_action = Greedy.choose_action
        elif self.action_context == PolicyMethods.SOFT_MAX:
            self.a.choose_action = PolicyMethods.SOFT_MAX

    def _build_action_state_computation(self) -> None:
        if self.action_context.action_methods == ActionMethods.NEURAL_NETWORK and \
        self.action_context.nn is not None:
            self.a.compute_actions_states = NeuralNetworkAction(self.action_context).compute_actions_states
        elif self.action_context.action_methods == ActionMethods.RANDOM:
            self.a.compute_actions_states = RandomActionSelection(self.action_context).compute_actions_states
        else:
            print('Something went wrong')

    def build_action(self) -> Action:
            
        self.a.make_exploration = SimpleExploration.make_exploration
        self._build_policy()
        self._build_action_state_computation()
        
        return self.a
    
