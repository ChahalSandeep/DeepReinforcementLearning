"""
example of openai cartpole
responsible for running episode for agent
"""
import gymnasium as gym


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    list_of_actions = []
    while True:
        action = env.action_space.sample()
        obs_, reward, done, _, info = env.step(action) #  obs reward termination truncated information
        total_reward += reward
        total_steps += 1
        list_of_actions.append(action)
        if done:
            break

    print("Episode done in %d steps and total reward is %.4f" % (total_steps, total_reward))
    print("actions: ", list_of_actions)
    env.close()





# env = gym.make('CartPole-v1', render_mode='human')
# env.reset()

# for i in range(100):
#     env.step(env.action_space.sample())
# env.close()