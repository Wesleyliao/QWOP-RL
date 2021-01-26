import os
import time

from stable_baselines.gail import generate_expert_traj

RECORD_PATH = os.path.join('pretrain', 'recording_1')
env = None


def dummy_expert(_obs):

    global env

    time.sleep(0.5)
    print(_obs)
    print(_obs.shape)

    return env.action_space.sample()


def human_expert(_obs):

    pass


def generate_obs(environment):
    global env
    env = environment
    generate_expert_traj(dummy_expert, RECORD_PATH, env, n_episodes=2)
