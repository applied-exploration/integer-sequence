"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py

Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random as rd
import collections
import numpy as np
from utils import flatten

# Exploration constant
c_PUCT = 1.38
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, state, n_actions, TreeEnv, action=None, parent=None, terminal=False, reward=0):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.parent = parent
        self.action = action
        self.state = state
        self.terminal = terminal
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}

        self.reward = reward

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        return self.child_Q + self.child_U

    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Choose action with highest score.
            possible_actions = self.TreeEnv.get_valid_actions(current.state)
            action_scores = [score if i in possible_actions else 0 for i, score in enumerate(current.child_action_score)]
            # print("\n\npossible_actions", possible_actions, "for state ", current.state)
            # print("\naction scores: ", action_scores)
            best_move = np.argmax(action_scores)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        if action not in self.children:
            # Obtain state following given action.
            new_state, reward, terminal, _ = self.TreeEnv.next_state(self.state, action)
            self.children[action] = MCTSNode(new_state, self.n_actions,
                                             self.TreeEnv,
                                             terminal=terminal,
                                             action=action, parent=self, reward=reward)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(flatten(self.state), self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "Node: action={}\033[0m".format(self.action)
        # node_string += "\n??? state:\n{}".format(flatten(self.state))
        # node_string += "\n??? N={}".format(self.N)
        # node_string += "\n??? score:\n{}".format(self.child_action_score)
        # node_string += "\n??? Q:\n{}".format(self.child_Q)
        # node_string += "\n??? P:\n{}".format(self.child_prior)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)

