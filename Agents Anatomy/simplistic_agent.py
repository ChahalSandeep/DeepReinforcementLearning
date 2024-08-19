"""
Simplistic python program
gives agent random rewards for limited number of steps regardless of agent's actions
"""
import random


class Environment:
    def __init__(self):
        self.steps_left = 10

    @staticmethod
    def get_observation():
        """ usually implemented some function of the internal state of environment
        :param
        :return: list[float, float, float]
        """
        return [0.0, 0.0, 0.0]

    @staticmethod
    def get_actions():
        """
        allows agent to query set of actions it can execute.
        normally does not change but some of them can become impossible
        here agent can carry out two actions encoded by 0 or 1
        :param
        :return: list[int, int
        """
        return [0, 1]

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
        if self.is_done():
            raise Exception("Environment is already done")
        self.steps_left -= 1
        return random.random()

class Agent:
    def __init__(self):
        # counter that keeps total rewards accumulated by agent during an episode
        self.total_reward = 0.0
        # actions that lead to the reward
        self.actions_taken = []

    def step(self, environment):
        """
        carries out a step in policy
        1. observes environment,
        2. makes decision about action ro take based on observation
        3. submits action to environments
        4. gets the reward for it
        taking an action and getting reward for that action which is added into reward accumulated by agent
        :param environment:
        :return:
        """
        current_observation = environment.get_observation()
        actions = environment.get_actions()
        current_action = random.choice(actions)
        reward = environment.action(current_action)
        self.actions_taken.append(current_action)
        self.total_reward += reward


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("total reward :%.4f " % agent.total_reward)
    print("actions taken that lead to reward :", agent.actions_taken)

