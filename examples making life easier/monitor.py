"""
the purpose of monitor class is to look at agents life inside the environment
this file contains example of adding monitor class to cartpole environment and agent
recording requires ffmpeg
"""

import gymnasium as gym
# from gymnasium.wrappers import

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    # env and directory it writes results to, the directory shouldn't exist
    env = gym.wrappers.RecordVideo(env, "recording")

    total_reward = 0.0
    total_steps = 0

    env.start_recording("videoname")
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs_, reward, done, _, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))

    env.stop_recording()
    env.close()
    env.env.close()