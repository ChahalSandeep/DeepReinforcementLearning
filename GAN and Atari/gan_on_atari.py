"""
This file contains GAN on Atari
some useful example of GAN's are
    image_quality_improvement,
    realistic image generation,
    feature learning
"""
# built-in packages
import random
import argparse

# third party packages
import gymnasium as gym
import ale_py  # need for atari breakouts
import numpy as np
import cv2
from torch import tensor, device, zeros, ones, FloatTensor
from torch.nn import BCELoss
from torch.optim import Adam
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# source packages
from simple_GAN import Generator, Discriminator, LATENT_VECTOR_SIZE

# CONSTANTS
IMAGE_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 10
SAVE_IMAGE_EVERY_ITER = 100

log = gym.logger
log.set_level(gym.logger.INFO)



class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input image
    -> resize image based on constants defined
    -> moving color channel for pytorch convolution digestible format
    -> normalizing image values from 0-255 to 0-1
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        # assert is observation space is Box type
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)

    def observation(self, observation):
        """
        :param self:
        :param observation:
        :return:
        """
        # resize image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # move color planes from last position to first position to meet pytorch convention of convolution layers
        # pytorch convolution layer conventions = (number of channels, height, width)
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        # followed by normalization by rescaling values between 0 and 1
        # new_obs.astype(np.float32) / 255
        return new_obs.astype(np.float32)


def iterate_batches(envs, batch_size=BATCH_SIZE):
    """
    we will use screenshots from Atari games played simultaneously by random agent
    :param envs:
    :param batch_size:
    :return:
    """
    batch = [e.reset()[0] for e in envs]

    env_gen = iter(lambda : random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _, info = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01: # required to prevent flickering of screen
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield tensor(batch_np)
            # yield FloatTensor(batch)
            batch.clear()
        if is_done:
            e.reset()

def main(device):
    # envs = []
    # for name in ('Breakout-v4', 'AirRaid-v0', 'Pong-v0'):
    #     print(name)
    #     envs.append(InputWrapper(gym.make(name)))
    #     print("done")
    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v4', 'AirRaid-v0', 'Pong-v0')]
    input_shape = envs[0].observation_space.shape
    net_discr = Discriminator(input_dim=input_shape).to(device)
    net_gener = Generator(output_dim=input_shape).to(device)

    objective = BCELoss() # loss function
    # optimizer for generator and discriminator
    # need to show both real 1 and fake 0 images to discriminator
    # after that we pass both fake and real and label 1's for all and update only generators weights
    gen_optimizer = Adam(params=net_gener.parameters(), lr=LEARNING_RATE)# ,etas=(0.5, 0.999))
    dis_optimizer = Adam(params=net_discr.parameters(), lr=LEARNING_RATE)# ,betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = ones(BATCH_SIZE, device=device)
    fake_labels_v = zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # we generate random vector and pass it to Generator network
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        # by training two times: to the true data sample in our batch vs generated
        # calls detach to avoid gradient flowing into generator
        # detach makes copy of tensor without connection to parents of operation
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar('gen_loss', np.mean(gen_losses), iter_no)
            writer.add_scalar('dis_loss', np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []

        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image('fake', vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image('real', vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true',help='enable cuda computation')
    args = parser.parse_args()
    device = device('cuda' if args.cuda else 'cpu')
    main(device)



