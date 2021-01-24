import click
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from game.env import QWOPEnv

TRAIN_TIME_STEPS = 60000
MODEL_PATH = "models/PPO2_MLP_v1"


def run_train():
    env = QWOPEnv()
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=TRAIN_TIME_STEPS)
    model.save(MODEL_PATH)


def run_test():
    env = QWOPEnv()
    model = PPO2.load(MODEL_PATH)
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
