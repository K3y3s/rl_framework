from nn.cnn import CNN
from env.environment import make_env
from agent.agent import AgentBuilder, ActionContext
from experiance.experiance import ExperienceBuffer, ExpSample
import torch

ENV_NAME = 'ALE/Pong-v5'
EPSILON = 0.05
POLICY = 'GREEDY'
ACTION_METHOD = 'RANDOM'
EXPERIENCE_BUFFER = 110
SYNCHRONIZATION = 0
LEARNING_RATE = 0.01
GAMMA = 0.1

if __name__ == '__main__':
    
    env = make_env(ENV_NAME)
    env.reset()
    

    nn = CNN(
        env.observation_space.shape,
        env.action_space.n
        )
    
    
    exp = ExperienceBuffer(EXPERIENCE_BUFFER)
    
    action_context = ActionContext(
        actions_number=env.action_space.n,
        action_methods=ACTION_METHOD,
        epsilon = EPSILON,
        nn = nn,
        policy= POLICY
    )
    
    
    agent = AgentBuilder(
        action_context,
        synchronization=SYNCHRONIZATION,
        env=env,
        gamma=GAMMA,
        loss_function=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        learning_rate=LEARNING_RATE
        ).build_agent()

    starting_state, _ = env.reset()
    
    for _ in range(100):
        action = agent.compute_action()
        exp_s = agent.make_step(action, starting_state)
        exp.append(exp_s)
        
        if exp_s.terminated:
            agent.env.reset()
        
    samples = exp.get_samples(15)
    
    agent.train_agent(samples=samples, target_model=nn)
    
    