import torch.nn as nn
import torch
from typing import Callable


class NN(nn.Module):
    """"
        Simple multilayer perceptron neural network.
        Hidden layers are fully connected
    """
    
    def __init__(self, 
        nb_of_state_variables:int, 
        nb_of_output_values:int,
        activation_functions:Callable,
        hidden_dims:tuple[int],
        add_biases:bool = False
        ):
        
        super().__init__()
        self.activation_functions = activation_functions
        self.nb_of_state_variables = nb_of_state_variables
        self.nb_of_output_values = nb_of_output_values
        
        self.input_layer = nn.Linear(
            self.nb_of_state_variables, 
            hidden_dims[0],
            bias= add_biases)
        
            
        self.hidden_layers = nn.ModuleList()
        
        if len(hidden_dims) > 1:
            for i in range(len(hidden_dims) - 1):
                hidden_layer = nn.Linear(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    bias=add_biases)    
                self.hidden_layers.append(hidden_layer)
        
        self.output_layer = nn.Linear(
            hidden_dims[-1],
            self.nb_of_output_values,
            bias = add_biases
        )
        
    def forward(self, state:torch.Tensor):
        
        x =  self.activation_functions(self.input_layer(state))

        for hidden_layer in self.hidden_layers:
            x = self.activation_functions(hidden_layer(x))
            
        return self.output_layer(x)