"""
Simplistic python program
gives agent random rewards for limited number of steps regardless of agent's actions
"""
import random


class Environment:
    def __init__(self):
        self.steps_left = 10

    @staticmethod
    def get_observation(self):
        """ usually implemented some function of the internal state of environment
        :param
        :return: list[float, float, float]
        """
        return [0.0, 0.0, 0.0]

    @staticmethod
    def get_actions(self):
        """
        allows agent to query set of actions it can execute.
        normally does not change but some of them can become impossible
        here agent can carry out two actions encoded by 0 or 1
        :param
        :return: list[int, int
        """
        return [0, 1]

    @staticmethod
    def is_done(self):
        """ signals end of episode"""
        return self.steps_left == 0

    def action(self, action):
        """
        central piece of environment
        does 2 things
        1. handles agent's action
        2. return reward for this action
        in this environment reward is random and action is discarded. Additionally,  we update the count of steps
        and refuse to continue the episodes that are over
        :param action:
        :return:
        """
        if self.is_done(self):
            raise Exception("Environment is already done")
        self.steps_left -= 1
        return random.random()