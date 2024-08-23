from copy import deepcopy
import numpy as np
import torch
from .agent_utils.synchronization import  SynchronizationBuilder, SyncContext
from .agent_utils.action_selection import ActionContext, ActionBuilder, Action
from experience.experience import ExpSample
from typing import Protocol, TypeVar, Any


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Env(Protocol):
    
    def step(action: ActType) -> tuple[ObsType, float, bool, bool | None, dict[str, Any] | None]:
            ...
            
class Optimizer(Protocol):
    
    def set() -> None:
        ...
        
    def zero_grad() -> None:
        ...
    

#TODO: change Module for online_model to something more generic with use of Protocol

class Agent:

    def __init__(self, 
                 online_model:torch.nn.Module,
                 gamma:float,
                 env:Env,
                 loss_function:callable,
                 optimizer:Optimizer,
                 learning_rate:float,
                 target_model: None
                 ) -> None:
        
        self.model = online_model
        self.env = env
        self.target_model = None
        self.gamma = gamma
        self._loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), self.learning_rate)
        self.target_model = target_model
        
        self.action: Action = None

    def synchronize(self, current_step:int) -> None:
        raise NotImplementedError
    
    def compute_action(self, *args, **kwargs) -> None:
        return self.action.compute_action(*args, **kwargs)
    
    def train_agent(self, samples:dict) -> None:
        self._train_agent(samples)
        
        
    def _train_agent(self, samples:dict) -> None:
        
        samples = self._prepare_data(samples)
      
        if self.target_model is not None:
            target_vals = self.target_model(samples['state']).max(1)[0].detach()
        else:
            target_vals = self.model(samples['state']).max(1)[0]
        
        target_vals[samples['terminated']] = 0.
        
        target_vals = samples['reward'] + self.gamma * target_vals
        
        predictions = self.model(samples['previous_state']) \
            .gather(1, samples['action'].unsqueeze(-1)).squeeze(-1)
        
        loss = self.loss_function(target_vals, predictions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def _prepare_data(self, samples:dict) -> dict:
        samples ={key: torch.tensor(np.array(item)) for key, item in samples.items()}
        self._reshape_data(samples)
        self._change_type(samples)
        return samples
        
    def _reshape_data(self, samples:dict) -> None:
        for reshape_name in ['state', 'previous_state']:
            if len(samples[reshape_name].shape) == 1:
                samples[reshape_name] = torch.reshape(samples[reshape_name], (-1, 1))      

    def _change_type(self, samples:dict) -> None:
        for name_to_float in ['state', 'previous_state', 'reward']:
            samples[name_to_float]  = samples[name_to_float].type(torch.float32)
            
        samples['action'] = samples['action'].type(torch.int64)
    
        
    def loss_function(self, target_vals:torch.tensor, predictions:torch.tensor) -> callable:
        #TODO: There are different loss function so at the end change it to the factory
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
        self.target_model = None if synchronization == 0 else deepcopy(self.nn)
        
        self.agent = Agent(
            online_model=self.nn, 
            env=env,
            gamma=gamma,
            loss_function=loss_function,
            optimizer=optimizer,
            learning_rate=learning_rate,
            target_model=self.target_model
            )
        
        self.agent.target_model = self.target_model

    def build_agent(self) -> Agent:
        sync_context = SyncContext(
            nn=self.nn, 
            synchronization=self.synchronize,
            nn_to_synchronize=self.target_model
            )
        self.agent.synchronize = SynchronizationBuilder(sync_context).build_synchronization()
        
        self.agent.action = ActionBuilder(self.action_context).build_action() 

        return self.agent
    
