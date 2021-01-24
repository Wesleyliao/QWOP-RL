import click
import tensorflow as tf
from stable_baselines import ACER

from game.env import QWOPEnv

TRAIN_TIME_STEPS = 80000
MODEL_PATH = "models/ACER_MLP_v1"


def run_train():

    # Define policy network
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[128, 128, 64])

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


@click.command()
@click.option('--train', default=False, is_flag=True, help='Run training')
@click.option('--test', default=False, is_flag=True, help='Run test')
def main(train, test):
    """Train and test an agent for QWOP."""

    if train:
        run_train()
    if test:
        run_test()

    if not test and not train:
        with click.Context(main) as ctx:
            click.echo(main.get_help(ctx))


if __name__ == '__main__':
    main()
