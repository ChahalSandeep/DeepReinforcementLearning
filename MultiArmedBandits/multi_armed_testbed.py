# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_armed_testbed.py
contains multi armed bandit test bud suite
"""

# Built-in/standard library

# Third/Other party packages
import numpy as np

# Owned/local source

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


class KArmedBandit(object):
    """
    Base class for k armed or multi armed bandit testbed
    """

    def __init__(self, **kwargs):
        """
        Inputs
        :param k: int (number of arms)
        :param epsilon: float (learning rate)
        :param n_iter: int (number od steps)
        """

        self.k = 10  # default number of arms
        self.action_set = np.zeros(self.k)  # step_count for each arm
        self.reward_set = np.zeros(self.k)  # Mean reward for each arm
        self.epsilon = 0.01  # initialize greedy selection
        self.n_iter = 10  # number od steps
        self.means = None  # initialized later in constructor
        self.curr_step = 0
        self.mean_reward = 0
        self.reward=np.zeros(self.n_iter)
        # todo check if maintaining count for each arm is helpful

        # update class variables
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # initialize values if not provided
        print("========================input========================")
        print("Note: if input is not provided default values are used.")

        if not isinstance(self.k, int): self.k = int(self.k)
        print("\tk/arms: {}".format(self.k))

        if not isinstance(self.epsilon, float): self.epsilon = float(self.epsilon)
        print("\tepsilon: {}".format(self.epsilon))

        self.reward = np.zeros(self.n_iter)  # todo re-wise this
        if not isinstance(self.n_iter, int): self.n_iter = int(self.n_iter)
        print("\tn_iter: {}".format(self.n_iter))

        if len(self.action_set) != self.k:
            # print("action_set != Number of arms : This would result in action_set with zeros"
            #       " which has equal number of arms")
            self.action_set = np.zeros(self.k)
        # else:
        #     print("using user specified action set")

        if len(self.reward_set) != self.k:
            # print("reward_set != Number of arms : This would result in result_set with zeros"
            #       " which has equal number of arms")
            self.reward_set = np.zeros(self.k)
        # else:
        #     print("using user specified reward set")

        # if type(self.action_set).__module__ != np.__name__:
        #     if isinstance(self.action_set, list):
        #         self.action_set = np.asarray(self.action_set)  # convert list to numpy array
        #     else: raise ValueError('action set must be list or numpy array')
        print("\taction_set: {}".format(self.action_set))

        # if type(self.reward_set).__module__ != np.__name__:
        #     if isinstance(self.reward_set, list):
        #         self.reward_set = np.asarray(self.reward_set)  # convert list to numpy array
        #     else: raise ValueError('reward set must be list or numpy array')
        print("\treward_set: {}".format(self.reward_set))

        if not (1 >= self.epsilon >= 0):
            raise ValueError("epsilon must be between 0 and 1")

        if self.means is None:
            self.means = np.random.normal(0, 1, self.k)  # drawing sample from random normal distribution
        elif len(self.means) != self.k:
            raise ValueError("len(means) != k/arms")
        else:
            print("using user specified means")

        if type(self.means).__module__ != np.__name__ :
            if isinstance(self.means, list):
                self.means = np.asarray(self.means)  # convert list to numpy array
            else: raise ValueError('action set must be list or numpy array')
        print("\tmeans: {}".format(self.means))

    def reset(self):
        raise NotImplementedError

    def pull(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


if __name__ == '__main__':
    armed_bandit_obj = KArmedBandit(k=2, epsilon=0.001, n_iter=5, epsilon_action="random")
    print("class variables: \n\t", armed_bandit_obj.__dict__)
