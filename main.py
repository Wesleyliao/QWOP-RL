import os

import click
import tensorflow as tf
from stable_baselines import ACER

from game.env import QWOPEnv
from pretrain import imitation_learning
from pretrain import recorder

TRAIN_TIME_STEPS = 100000
RECORD_PATH = os.path.join('pretrain', 'human_try1')
MODEL_PATH = os.path.join('models', 'ACER_MLP_v2')


def define_model():

    # Define policy network
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 128])

    # Initialize env and model
    env = QWOPEnv()
    model = ACER(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        n_cpu_tf_sess=5,
        replay_start=40,
        verbose=1,
    )

    return model


def run_train():

    model = define_model()

    # Train and save
    model.learn(total_timesteps=TRAIN_TIME_STEPS)
    model.save(MODEL_PATH)


def continue_learning():

    env = QWOPEnv()

    # Load model
    model = ACER.load(MODEL_PATH)
    model.set_env(env)

    # Train and save
    model.learn(total_timesteps=TRAIN_TIME_STEPS)
    model.save(MODEL_PATH)


def run_test():

    # Initialize env and model
    env = QWOPEnv()
    model = ACER.load(MODEL_PATH)

    # Run test
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


def run_record():

    env = QWOPEnv()
    recorder.generate_obs(env)


@click.command()
@click.option('--train', default=False, is_flag=True, help='Run training')
@click.option('--test', default=False, is_flag=True, help='Run test')
@click.option(
    '--record',
    default=False,
    is_flag=True,
    help='Record observations for pretraining',
)
@click.option(
    '--imitate', default=False, is_flag=True, help='Train agent from recordings'
)
def main(train, test, record, imitate):
    """Train and test an agent for QWOP."""

    if train:
        run_train()
    if test:
        run_test()

    if record:
        run_record()

    if imitate:
        model = define_model()
        imitation_learning.imitate(model, MODEL_PATH)

    if not (test or train or record or imitate):
        with click.Context(main) as ctx:
            click.echo(main.get_help(ctx))


if __name__ == '__main__':
    main()
