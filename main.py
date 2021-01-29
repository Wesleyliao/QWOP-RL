import os
import time

import click
import tensorflow as tf
from stable_baselines import ACER
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import SubprocVecEnv

from game.env import QWOPEnv
from pretrain import imitation_learning
from pretrain import recorder

MODEL_NAME = 'ACER_MLP_V3'
TRAIN_TIME_STEPS = 150000
RECORD_PATH = os.path.join('pretrain', 'human_try1')
MODEL_PATH = os.path.join('models', MODEL_NAME)

checkpoint_callback = CheckpointCallback(
    save_freq=TRAIN_TIME_STEPS / 10, save_path='./logs/', name_prefix=MODEL_NAME
)


def define_model():

    # Define policy network
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 128])

    # Initialize env and model
    env = QWOPEnv()
    model = ACER(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        replay_start=40,
        verbose=1,
    )

    return model


def run_train(model_path=MODEL_PATH):

    if os.path.isfile(model_path + '.zip'):
        print('--- Training from existing model', model_path, '---')
        model = ACER.load(model_path)
        env = SubprocVecEnv([lambda: QWOPEnv()])
        model.set_env(env)
    else:
        model = define_model()

    # Train and save
    t = time.time()

    model.learn(total_timesteps=TRAIN_TIME_STEPS, callback=checkpoint_callback)
    model.save(MODEL_PATH)

    print(f"Trained {TRAIN_TIME_STEPS} steps in {time.time()-t} seconds.")


def run_test():

    # Initialize env and model
    env = QWOPEnv()
    model = ACER.load(MODEL_PATH)

    # Run test
    t = time.time()
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    input(f"Test run complete in {time.time()-t} seconds. Press Enter to exit.")


@click.command()
@click.option(
    '--train',
    default=False,
    is_flag=True,
    help='Run training; will train from existing model if path exists',
)
@click.option('--test', default=False, is_flag=True, help='Run test')
@click.option(
    '--record',
    default=False,
    is_flag=True,
    help='Record observations for pretraining',
)
@click.option(
    '--imitate',
    default=False,
    is_flag=True,
    help='Train agent from recordings; will use existing model if path exists',
)
def main(train, test, record, imitate):
    """Train and test an agent for QWOP."""

    if train:
        run_train()
    if test:
        run_test()

    if record:
        env = QWOPEnv()
        recorder.generate_obs(env, RECORD_PATH)

    if imitate:
        model = define_model()
        imitation_learning.imitate(model, RECORD_PATH, MODEL_PATH)

    if not (test or train or record or imitate):
        with click.Context(main) as ctx:
            click.echo(main.get_help(ctx))


if __name__ == '__main__':
    main()
