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
    epsilon_decay: int | None = None # after how many steps epsilon should be set to epsilon_value
    epsilon_decay_value: float | None = None # what is a value of an epsilon after epsilon_decay 
    

    def __post_init__(self):
        self.action_methods =ActionMethods[self.action_methods]
        self.policy = PolicyMethods[self.policy]
        
        if (self.epsilon_decay is None) ^ (self.epsilon_decay_value is None):
            raise ValueError("Set both values epsilon decay and epsilon_decay_value")
        
############################################ Policy

class Policy(ABC):

    @abstractmethod
    def choose_action(self, action_state_values):
        raise NotImplementedError

class Greedy(Policy):
    
    def choose_action(self, action_state_values:Tensor):
        return argmax(action_state_values).item()
    
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
        self.current_step = 0
        self._trigger_decay = True
        self.epsilon = action_context.epsilon

    @property
    def trigger_decay(self) -> bool:
        return self._trigger_decay
    
    @trigger_decay.setter
    def trigger_decay(self, trigger:bool) -> None:
        self._trigger_decay = trigger
        self.current_step = 0

    def make_exploration() -> int:
        raise NotImplementedError

    def compute_actions_states() -> int | list[float]:
        raise NotImplementedError

    def choose_action(self, action_state_values: list[float]) -> int:
        raise NotImplementedError

    def _decay_epsilon(self) -> float:
        if self.action_context.epsilon_decay <= self.current_step:
                epsilon = 0.
            
        else:
                eps_diff = self.action_context.epsilon - self.action_context.epsilon_decay_value
                epsilon = self.action_context.epsilon -  eps_diff/self.action_context.epsilon_decay * self.current_step
        
        return epsilon
    
    def _compute_epsilon(self) -> float:
        
        if self.action_context.epsilon_decay is not None and self._trigger_decay:
            epsilon = self._decay_epsilon()
                
        else:
            epsilon = self.action_context.epsilon
    
        self.epsilon = epsilon
        return epsilon
    
    def compute_action(self, states = None) -> int | float:
        
        epsilon = self._compute_epsilon()
        
        if np.random.random() < epsilon:
            action_number = self.make_exploration(self.action_context.actions_number)
        
        else:
            actions_vals = self.compute_actions_states(states)

            if not isinstance(actions_vals, int):
                action_number = self.choose_action(actions_vals)
            else:
                action_number = actions_vals

        self.current_step +=1
        
        return action_number


class ActionBuilder:
    def __init__(self, action_context:ActionContext):
        self.a = Action(action_context)
        self.action_context = action_context

    def _build_policy(self) -> None:
        if self.action_context.policy == PolicyMethods.GREEDY:
            self.a.choose_action = Greedy().choose_action
        elif self.action_context.policy == PolicyMethods.SOFT_MAX:
            self.a.choose_action = SoftMax().choose_action

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
    
