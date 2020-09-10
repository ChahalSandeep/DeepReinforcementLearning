# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_armed_testbed.py
contains multi armed bandit test bud suite
"""

# Built-in/standard library

# Third/Other party packages
import numpy as np
import matplotlib.pyplot as plt

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
        self.action_set = np.zeros(self.k)  # default action set  todo check if this is right place
        self.epsilon = 0.01  # initialize greedy selection
        self.n_iter = 10  # number od steps
        self.optimal = 0  # reward

        # update class variables
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # initialize values if not provided
        print("========================input========================")
        print("Note: if input is not provided default values are used.")

        if not isinstance(self.k, int): self.k = int(self.k)
        print("\tk/arms: {}".format(self.k))

        if not isinstance(self.epsilon, float): self.epsilon = float(self.epsilon)
        print("\tepsilon: {}".format(self.epsilon))

        if type(self.action_set).__module__ != np.__name__:
            if isinstance(self.action_set, list):
                # todo convert list to numpy array
                ...
            else: raise ValueError('action set must be list or numpy array')

        print("\taction_set: {}".format(self.action_set))

        # initialize number of iterations
        if not isinstance(self.n_iter, int): self.n_iter = int(self.n_iter)
        print("\tn_iter: {}".format(self.n_iter))

    def reset(self):
        raise NotImplementedError

    def pull(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


if __name__ == '__main__':
    armed_bandit_obj = KArmedBandit(k=2, epsilon=0.001, n_iter=5, epsilon_action="random")
    print("class variables: \n\t", armed_bandit_obj.__dict__)
