"""
wrappers are helpful if you want to extend OpenAi gym functionality
example:
1. Accumulate observation overtime in buffer.
    use: provide agent with last N observation
    e.g. dynamic computer games
2. preprocess state or observation before providing to agent
    use: image pixels provide agent with processed image
    e.g. road segmented to uvs for example
3. accumulate reward
    use: if we get reward at the end of episode we may want to backprop.
"""

import gymnasium as gym
import random
from typing import TypeVar

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    """
    intervening with actions with probability of 10% replacing the current action with random action
    """
    def __init__(self, env, epsilon=0.1):
        # Initialize wrapper by calling parents __init__ method and saving epsilon as a probability
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        """
        replacing action method of parent class by adding probability
        :param action: float (probability of replacing current action by random)
        :return: action
        """
        if random.random() < self.epsilon:
            print("random")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make('CartPole-v1'))
    obs = env.reset()
    total_reward = 0
    total_steps = 0
    while True:
        obs_, reward, done, _, info = env.step(0)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps and total reward is %.4f" % (total_steps, total_reward))



