"""
RL - cross entropy (Model free, Policy Based and on policy)
Strengths: Simplicity and Good Convergence
for problems that don't require complex, multistep policies to be learned and discovered
that has short episodes with frequent rewards.
can be used as a part of larger system or by its own

policy = probability distribution over action space similar to classification problem
cross entropy throws away bad episode and train on better episodes

Algorithm:
1. Play N number of episodes using current model and environment
2. Calculate total reward for every episode and decide reward boundary. Usually use some percentile for all rewards
    such as 50th or 70th
3. Throw away all episodes with reward below boundary
4. Train on remaining episodes using observation as input and actions as output
5. Repeat 1-4 until we are satisfied
"""
# built-in packages
from collections import namedtuple

# third-party packages
from torch import nn
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import gymnasium as gym
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# source project packages
from nn_archs import CrossEntropyNeuralNet as CrossNet

# CONSTANTS
HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE_TO_SELECT = 70
REWARD_THRESHOLD = 999


# defining helper classes using collection

# single episode stored as total un-discounted reward and collection of Episode Steps
# Note that will not be using discounted reward anyway reason? given below from Sutton
# refer to 31:37 https://www.youtube.com/watch?v=uGOGvALFWbo
Episode = namedtuple('Episode', field_names=['episode_reward', 'episode_steps'])

# represents single step agent makes in episode and stores observation and action agent completed.
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# function that generates batches with episodes

def iterate_batches(env, net, batch_size):
    """

    :param env: enc class instance from gym library
    :param net: defined neural network (nn.module)
    :param batch_size: count of episodes per iteration
    :return:
    """
    g_batch = [] # accumulates batch (list of episode instances)
    episode_rewards = 0.0 # reward for current episode
    episode_steps = [] # lost of steps in episode
    obs_initial = env.reset() # reset observation to first observation
    obs = obs_initial[0] # also gives a dictionary with observation
    g_softmax = nn.Softmax(dim=1) # used to convert networks output to probabilities of actions

    # at every iteration we convert observation to pytorch tensor and pass to network
    # and get action probabilities.

    while True:
        # take observation
        obs_v = FloatTensor([obs])
        # use simple neural net architecture feed in our observation and then use softmax on action space
        action_prob_v = g_softmax(net(obs_v))
        act_probs = action_prob_v.data.numpy()[0]

        # we have action probability distribution we can now choose action and get next observation
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _, info = env.step(action)

        episode_rewards += reward # append reward to reward of episode
        episode_steps.append(EpisodeStep(observation=obs, action=action)) # add observation and reward

        # check if the current episode is over (when cart pole stick falls down)
        if is_done:
            # append finalized episode to batch
            g_batch.append(Episode(episode_rewards, episode_steps=episode_steps)) # add episode reward and number of steps
            # reset environment and episodic variables
            episode_rewards = 0.0
            episode_steps = []
            next_obs = env.reset()[0]  # gets rid of dictionary in the tuple
            # if batch has reached desired episode count return it to caller for processing using yield
            if len(g_batch) == batch_size:
                # generator function control is transferred to outer iteration loop and then continued after yield
                yield g_batch  # runs for 16 episodes
                # clean up batch
                g_batch = []

        obs = next_obs

# core function of cross entropy method
def filter_batch(g_batch, percentile):
    """
    core function of cross entropy method
    1. given batch of episodes and percentile value, calculates boundary reward used to select elite episode & then trained
    2. we use numpy percentile func to obtain boundary reward (is list of values and desired percentile)
    3. calculate mean reward for monitoring
    :param g_batch:
    :param percentile:
    :return:
    """
    rewards = list(map(lambda s: s.episode_reward, g_batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_observations = []
    train_actions = []

    # for every episode in batch check if the episode has higher total reward than boundary
    # if so then populate observations and actions to train on

    for reward, steps in g_batch:
        if reward < reward_bound:
                    continue
        train_observations.extend(map(lambda step: step.observation, steps))
        train_actions.extend(map(lambda step: step.action, steps))
        #     for g_episode in g_batch:
        #         if g_episode.episode_reward < reward_bound:
        #             continue
        #         train_observations.extend(lambda step: step.observation, g_episode.episode_reward)

    # convert observation and actions into tensors
    # return tuple of obs, action, boundary of reward and mean reward
    train_observations_v = FloatTensor(train_observations)
    train_actions_v = FloatTensor(train_actions)
    # bound and mean to check performance of agent
    return train_observations_v, train_actions_v, reward_bound, reward_mean


def main():
    env = gym.make('CartPole-v0', render_mode="rgb_array") # create env
    obs_size = env.observation_space.shape[0] # get observation size
    n_actions = env.action_space.n # get number of actions
    net = CrossNet(obs_size, HIDDEN_SIZE , n_actions) # object for our nn
    obj = nn.CrossEntropyLoss() # define loss function
    optimizer =optim.Adam(params=net.parameters(), lr=0.001) # define optimizer
    writer = SummaryWriter('logs/') # initialize writer

    env = gym.wrappers.RecordVideo(env, "recording")
    env.start_recording("cartpole_video_game")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE_TO_SELECT)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = obj(action_scores_v, acts_v.type(dtype=LongTensor)) # torch function has issues with float type
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar('loss', loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > REWARD_THRESHOLD:
            print("Solved!")
            env.stop_recording()
            env.close()
            env.env.close()
            break
    writer.close()
    # env.stop_recording()
    # env.close()
    # env.env.close()


if __name__ == '__main__':
    main()




