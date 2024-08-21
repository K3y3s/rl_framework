from nn.nn import NN
from agent.agent import AgentBuilder, ActionContext
from experience.experience import ExperienceBuffer
import torch
import matplotlib.pyplot as plt
from copy import deepcopy


ENV_NAME = 'FrozenLake-v1'
EPSILON = 1
EPSILON_DECAY = 100
EPSILON_DECAY_VALUE = 0 
POLICY = 'GREEDY'
ACTION_METHOD = 'NEURAL_NETWORK'
EXPERIENCE_BUFFER = 6
SAMPLE_SIZE = 4
SYNCHRONIZATION_RATE = 2
LEARNING_RATE = 0.01
GAMMA = 0.95
HIDDEN_DIMENSION = (8, 8,8, 4, 4)
A_FUNCTION = torch.nn.ReLU()
LOSS_F=torch.nn.MSELoss()
OPTIMIZER=torch.optim.Adam
STEPS_TO_PLAY = 1000


from tests.utils.test_env import TestEnv
env = TestEnv()

nn = NN(
    nb_of_state_variables = 1,
    nb_of_output_values=4,
    activation_functions= A_FUNCTION,
    hidden_dims=HIDDEN_DIMENSION,
    add_biases=True
    )

exp = ExperienceBuffer(
    size_of_buffer=EXPERIENCE_BUFFER, 
    sampling_method='SIMPLE_RANDOM'
    )

action_context = ActionContext(
    actions_number=4,
    action_methods=ACTION_METHOD,
    epsilon = EPSILON,
    nn = nn,
    policy= POLICY,
    epsilon_decay=EPSILON_DECAY,
    epsilon_decay_value=EPSILON_DECAY_VALUE
)

agent = AgentBuilder(
    action_context,
    env=env,
    gamma=GAMMA,
    loss_function=LOSS_F,
    optimizer=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    synchronization = SYNCHRONIZATION_RATE # after how many step synchronization will be made
    ).build_agent()

agent.trigger_decay = False

counter = 0
episode_steps = 0
trigger = True

episode_steps_all = []

reward_every_100_episodes = []
reward = []

starting_state, _ = env.reset()

while counter <= STEPS_TO_PLAY:
   
    starting_state = torch.tensor([starting_state], dtype=torch.float)
    starting_state.unsqueeze(0)
    
    action = agent.compute_action(starting_state)
    print(action)
    exp_s = agent.make_step(action, starting_state[0].item())
    exp.append(exp_s)
    starting_state = exp_s.state
    
    if exp_s.terminated:
        print(exp_s)
        reward.append(exp_s.reward)
        if exp_s.reward == 1:
            if trigger:
                agent.trigger_decay = True
                trigger = False
            print(f'Epsilon {agent.action.epsilon}')
            print(f'Episode ended after steps: {episode_steps}, Reward = {exp_s.reward}')
         
        episode_steps_all.append(episode_steps)
        episode_steps = 0
        starting_state, _ = agent.env.reset()
        
    
    # train agent
    if len(exp) >= EXPERIENCE_BUFFER:
        samples = exp.get_samples(SAMPLE_SIZE)
        agent.train_agent(samples)
        
    # synchronization
    agent.synchronize(counter)
    
    counter +=1
    episode_steps +=1
    
    
from torch import argmax
for i in range(4):
    print(f'{i}: {agent.model(torch.tensor([i], dtype=torch.float))}, \
          {argmax(agent.model(torch.tensor([i], dtype=torch.float)))}')