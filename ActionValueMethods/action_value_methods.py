# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_value_methods.py

"""

# Built-in/standard library

# Third/Other party packages
import numpy as np

# Owned/local source
from MultiArmedBandits.multi_armed_testbed import KArmedBandit

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


class GreedyKArmedBandit(KArmedBandit):
    """
    """
    def __init__(self, **kwargs):
        """
        Inputs
        :param k: int (number of arms)
        :param epsilon: float (learning rate)
        :param n_iter: int (number od steps)
        """
        super().__init__(**kwargs)
        self.mean_reward_overtime = []

    def pull(self):
        # Generate random number between 0 and 1
        generated_value = np.random.rand()

        # if no epsilon then makes all the choices as random
        if self.epsilon == 0 or self.n_iter == 0:
            curr_action = np.random.choice(self.k)
        # if generated value is less than epsilon then takes random action
        elif generated_value < self.epsilon:
            curr_action = np.random.choice(self.k)
        # take greedy action
        else:
            curr_action = np.argmax(self.mean_reward_set)

        # get 1 reward according to action from means
        curr_reward = np.random.normal(self.means[curr_action], 1)

        # update step count
        self.curr_step += 1
        # update total_reward
        self.total_reward = self.total_reward + curr_reward
        # update total reward for arm selected
        self.total_each_arm_reward[curr_action] = self.total_each_arm_reward[curr_action] + curr_reward
        # update the action count for arm
        self.action_set[curr_action] = self.action_set[curr_action] + 1
        # derive the mean reward so far for the arm
        self.mean_reward_set[curr_action] = self.total_each_arm_reward[curr_action] / self.action_set[curr_action]

        # update total mean reward
        self.mean_reward = self.total_reward / self.curr_step

    def run(self, print_info=False):
        self.mean_reward_overtime.append(0)
        for n in range(self.n_iter):
            self.pull()
            if print_info:
                print("actual = ", self.means)
                print("step: {}, mean_reward = {}, reward_set = {}".format(n + 1, self.mean_reward, self.mean_reward_set))
                print("mean reward ", self.mean_reward)

            self.mean_reward_overtime.append(self.mean_reward)

    def reset(self):
        raise NotImplementedError


if __name__ == '__main__':
    armed_bandit_obj = GreedyKArmedBandit(k=5, epsilon=0.2, n_iter=1000, epsilon_action="random")
    print("class variables: \n\t", armed_bandit_obj.__dict__)
    armed_bandit_obj.run(print_info=True)
