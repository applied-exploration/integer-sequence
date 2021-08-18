from .mcts import MCTS

import random as rd
import numpy as np
from utils import flatten

def execute_episode(agent_netw, num_simulations, TreeEnv):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv)
    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_leaf()

    flattened_state = flatten(first_node.state)
    converted_state = np.array(flattened_state).astype(np.float32)
    observations_for_states = TreeEnv.get_obs_for_states(converted_state)

    
    probs, vals = agent_netw.step(np.array([observations_for_states]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.terminal:
            break

    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    # ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
    #        in range(len(mcts.rewards))]
    ret = np.cumsum(mcts.rewards[::-1])[::-1]
    # print("ret")
    # print(ret)


        

    total_rew = np.sum(mcts.rewards)
    # obs = np.concatenate(mcts.obs)
    obs = np.array(mcts.obs)
    searches_pi = np.array(mcts.searches_pi)
    
    # print("OBS")
    # print(mcts.obs)
    # print(obs.shape)
    # print(mcts.searches_pi)
    # print(ret.shape)
    # print(total_rew.shape)

    return (obs, searches_pi, ret, total_rew, mcts.root.state)