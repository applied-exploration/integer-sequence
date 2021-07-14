from collections import deque
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils import flatten
import torch

def train_env(env, agents, max_t, num_episodes, scores_window=100, print_every = 20, save_states_every = 0, experiment_num=0):
    score_history = []
    state_history = []
    scores_deque = deque(score_history[-scores_window:], maxlen=scores_window)
    last_running_mean = float('-inf')

    for episode in range(num_episodes):
        states = env.reset()
        scores = np.zeros(len(agents))

        for i in range(max_t):
            if not save_states_every < 1 and episode % save_states_every == 0:
                state_history.append(states)


            actions = [agent.act(state) for agent, state in zip(agents, states)]

            next_states, rewards, done = env.step(actions)
            [agent.step(state, action, reward, next_state, done) for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, next_states)]

            scores += rewards

            states = next_states
            if done == True:
                break

        scores_deque.append(scores)
        score_history.append(scores)
        returns_in_episode = np.mean(scores)

        [agent.reset() for agent in agents]
        if episode > scores_window:
            if np.mean(scores_deque) > last_running_mean:
                    # print("")
                    # print('Last {} was better, going to save it'.format(scores_window))
                    for num_agent, agent in enumerate(agents):
                        agent.save(experiment_num, num_agent)

                    last_running_mean = np.mean(scores_deque)
     
            print("\r", 'Total score (averaged over agents) {} episode: {} | \tAvarage in last {} is {}'.format(episode, returns_in_episode, scores_window, np.mean(scores_deque)), end="")


    return score_history, state_history
