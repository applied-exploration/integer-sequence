"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(os.path.abspath(os.path.join('..', 'function-gen')))

import time
import numpy as np
import matplotlib.pyplot as plt


from IPython.display import display


from .trainer import Trainer
from .policy import IntegerPolicy
from .replay_memory import ReplayMemory
# from hill_climbing_env import HillClimbingEnv
from .MCTS import MCTS

# from lang import load_data_int_seq
# from models.rl.env import IntegerSequenceEnv
from .execute_episode import execute_episode

from utils import flatten

import sys, os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append(os.path.abspath(os.path.join('..', '..', '..', 'rl')))

from rl.env import decode_with_lang


def train_mcts(env, num_epochs, policy=None, test=False):
    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]

    trainer = Trainer(lambda: IntegerPolicy(n_obs, 20, n_actions) if policy is None else policy)
    network = trainer.step_model

    mem = ReplayMemory(200,
                    { "ob": np.long,
                        "pi": np.float32,
                        "return": np.float32},
                    { "ob": [n_obs],
                        "pi": [n_actions],
                        "return": []})

    value_losses = []
    policy_losses = []
    total_rews = []

    for i in range(num_epochs):
        if i % 50 == 0: 
            fig, axs = plt.subplots(3, 1)
   
            axs[0].plot(value_losses)
            axs[0].set_xlabel("epochs")
            axs[0].set_ylabel("value loss")
 
            axs[1].plot(policy_losses)
            axs[1].set_xlabel("epochs")
            axs[1].set_ylabel("policy loss")
            
            total_rew = test_agent(i, env, network)
            total_rews.append(total_rew)

            axs[2].plot(total_rews)
            axs[2].set_xlabel("epochs")
            axs[2].set_ylabel("testing loss")

            plt.savefig("training_{}.jpg".format(num_epochs))

            if test: break


        obs, pis, returns, tot_reward, done_state = execute_episode(network, 64, env, iteration=i)   
        
        mem.add_all({"ob": obs, "pi": pis, "return": returns})
        batch = mem.get_minibatch(batch_size=None)

        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])

        value_losses.append(min(vl.item(), 1))
        policy_losses.append(pl.item())

            
            
            
def test_agent(iteration, env, network):
    print("Testing Agent")
    
    _, __, ___, total_reward, ____  = execute_episode(network, 64, env, log = log, test = True, iteration = iteration)  
    
    return total_reward

# def test_agent(iteration, env, network):
#     print("Testing Agent")
#     test_env = env
#     total_rew = 0
#     state, reward, done, _ = test_env.reset()
#     step_idx = 0
#     while not done:
#         log(test_env, iteration, step_idx, total_rew)
#         p, _ = network.step(np.array([flatten(state)]).astype(np.float32))

#         action = np.argmax(p)
#         state, reward, done, _ = test_env.step(action)
#         step_idx+=1
#         total_rew += reward
#     log(test_env, iteration, step_idx, total_rew)
    
#     return total_rew
        
        
def log(obs, iteration, step_idx, total_rew, env, action = None):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    print(f"Action: {action}")
    if action is not None:
        print(f"State:")
        print(f"{obs[-1]}")
        decoded_obs = decode_with_lang(env.output_lang, obs[-1][:9]) 
        print(f"{decoded_obs}")
    # test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")
    

