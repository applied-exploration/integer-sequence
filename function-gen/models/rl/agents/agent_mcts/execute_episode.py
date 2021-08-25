from .MCTS import MCTS

import random as rd
import numpy as np
from utils import flatten


def execute_episode(agent_netw, num_simulations, TreeEnv, log = None, test = False, iteration=0):
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

    if log:
        step_idx = 0
        reward = 0
        action = None
    
    while True:
        if log: 
            # log(np.array(mcts.obs), iteration, step_idx, reward, action)
            step_idx += 1
        # if test: mcts.root.print_tree()
        
        if not test: mcts.root.inject_noise()
        current_simulations = mcts.root.N

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()
        
        
        action = mcts.pick_action()
        reward = mcts.take_action(action)

        if mcts.root.terminal:
            mcts.save_last_ob()
            break
    
    # mcts.root.print_tree()
    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    # ret = [TreeEnv.get_return(mcts.root.state, mcts.root.depth) for _
    #        in range(len(mcts.rewards))]
    ret = np.cumsum(mcts.rewards[::-1])[::-1]
        

    total_rew = np.sum(mcts.rewards)
    obs = np.array(mcts.obs)
    searches_pi = np.array(mcts.searches_pi)
    
    if log: 
        log(obs, iteration, step_idx, total_rew, TreeEnv, action)
    
    return (obs, searches_pi, ret, total_rew, mcts.root.state)