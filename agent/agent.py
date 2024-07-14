from copy import deepcopy
import numpy as np
from torch.nn import Module
from torch import tensor
from agent.agent_utils.synchronization import SyncContex, SynchronizationBuilder
from agent.agent_utils.action_selection import ActionContext, ActionBuilder, Action
from experiance.experiance import ExpSample
from typing import Protocol, TypeVar, Any

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Env(Protocol):
    
    def step(action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
            ...
            
class Optimizer(Protocol):
    
    def set() -> None:
        ...
        
    def zero_grad() -> None:
        ...
    

#TODO: change Module to something more generic with use of Protocol

class Agent:

    def __init__(self, 
                 online_model:Module,
                 gamma:float,
                 env:Env,
                 loss_function:callable,
                 optimizer:Optimizer,
                 learning_rate:float
                 ) -> None:
        
        self.model = online_model
        self.env = env
        self.model_to_synchronize = None
        self.gamma = gamma
        self._loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), self.learning_rate)
        
        self.action: Action = None

    def synchronize(self, current_step:int) -> None:
        raise NotImplementedError
    
    def compute_action(self, *args, **kwargs) -> None:
        return self.action.compute_action(*args, **kwargs)
    
    def train_agent(self, samples:dict, target_model:Module) -> None:
        
        samples = {key:tensor(np.array(item)) for key, item in samples.items()}
        
        target_vals = target_model(samples['state']).max(1)[0].detach()
        target_vals[samples['terminated']] = 0.
        
        target_vals = samples['reward'] + self.gamma * target_vals
        predictions = self.model(samples['previous_state']).gather(1, samples['action'].unsqueeze(-1))
        
        loss = self.loss_function(target_vals, predictions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        
    def loss_function(self, target_vals:tensor, predictions:tensor) -> callable:
        #There are different loss function so at the end chage it to the factory
        return self._loss_function(predictions, target_vals)
        
        
    def make_step(self, action:ActType, prev_state:ObsType) -> ExpSample:
        new_state, reward, is_done, _, _ = self.env.step(action)
        return ExpSample(
            state = new_state,
            reward= reward,
            terminated=is_done,
            previous_state=prev_state,
            action=action
            )
    
class AgentBuilder:
    
    def __init__(self, 
                 action_context:ActionContext,
                 env:Env,
                 gamma:float,
                 loss_function:callable,
                 optimizer: Optimizer,
                 learning_rate:float,
                 synchronization:int = 0) -> None:
        
        self.action_context = action_context
        self.nn = action_context.nn
        self.synchronize = synchronization
        self.nn_to_synchronize = None if synchronization == 0 else deepcopy(self.nn)
        
        self.agent = Agent(
            online_model=self.nn, 
            env=env,
            gamma=gamma,
            loss_function=loss_function,
            optimizer=optimizer,
            learning_rate=learning_rate
            )
        
        self.agent.nn_to_synchronize = self.nn_to_synchronize

    def build_agent(self) -> Agent:

        sync_context = SyncContex(
            nn=self.nn, 
            synchronization=self.synchronize,
            nn_to_synchornize=self.nn_to_synchronize
            )

        self.agent.synchronize = SynchronizationBuilder(sync_context).build_synchronization()
        
        self.agent.action = ActionBuilder(self.action_context).build_action() 

        return self.agent
    
