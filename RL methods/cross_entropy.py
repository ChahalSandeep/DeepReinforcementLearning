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
import numpy as np

# source project packages
from nn_archs import CrossEntropyNeuralNet as cross_net


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
    obs = env.reset() # reset observation to first observation
    g_softmax = nn.Softmax(dim=1) # used to convert networks output to probabilities of actions

    # at every iteration we convert observation to pytorch tensor and pass to network
    # and get action probabilities.

    while True:
        # take observation
        obs_v = FloatTensor([obs])
        # use simple neural net architecture feed in our observation and then use softmax on action space
        action_prob_v = g_softmax(cross_net(obs_v))
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
            next_obs = env.reset()
            # if batch has reached desired episode count return it to caller for processing using yield
            if len(g_batch) == batch_size:
                # generator function control is transferred to outer iteration loop and then continued after yield
                yield g_batch
                # clean up batch
                g_batch = []

        obs = next_obs




