import os
import time

import click
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.deepq.policies import FeedForwardPolicy

from game.env import QWOPEnv
from pretrain import imitation_learning
from pretrain import recorder

# Training parameters
MODEL_NAME = 'DQN_imitateACER'
EXPLORATION_FRACTION = 0.2
LEARNING_STARTS = 1000
EXPLORATION_INITIAL_EPS = 0.01
EXPLORATION_FINAL_EPS = 0.01
BUFFER_SIZE = 50000
BATCH_SIZE = 32
TRAIN_FREQ = 4
LEARNING_RATE = 0.000002
TRAIN_TIME_STEPS = 200000
MODEL_PATH = os.path.join('models', MODEL_NAME)
TENSORBOARD_PATH = './tensorboard/'

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path='./logs/', name_prefix=MODEL_NAME
)

# Imitation learning parameters
RECORD_PATH = os.path.join('pretrain', 'acer_114hr_100episodes')
N_EPISODES = 100
N_EPOCHS = 200


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            *args,
            **kwargs,
            layers=[256, 128],
            layer_norm=True,
            feature_extraction="mlp",
        )


def get_new_model():

    # Initialize env and model
    env = QWOPEnv()
    model = DQN(
        CustomDQNPolicy,
        env,
        prioritized_replay=True,
        verbose=1,
        tensorboard_log=TENSORBOARD_PATH,
    )

    return model


def get_existing_model(model_path):

    print('--- Training from existing model', model_path, '---')

    # Load model
    model = DQN.load(model_path, tensorboard_log=TENSORBOARD_PATH)

    # Set environment
    env = QWOPEnv()  # SubprocVecEnv([lambda: QWOPEnv()])
    model.set_env(env)

    return model


def get_model(model_path):

    if os.path.isfile(model_path + '.zip'):
        model = get_existing_model(model_path)
    else:
        model = get_new_model()

    return model


def run_train(model_path=MODEL_PATH):

    model = get_model(model_path)
    model.learning_rate = LEARNING_RATE
    model.learning_starts = LEARNING_STARTS
    model.exploration_initial_eps = EXPLORATION_INITIAL_EPS
    model.exploration_final_eps = EXPLORATION_FINAL_EPS
    model.buffer_size = BUFFER_SIZE
    model.batch_size = BATCH_SIZE
    model.train_freq = TRAIN_FREQ

    # Train and save
    t = time.time()

    model.learn(
        total_timesteps=TRAIN_TIME_STEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )
    model.save(MODEL_PATH)

    print(f"Trained {TRAIN_TIME_STEPS} steps in {time.time()-t} seconds.")


def run_test():

    # Initialize env and model
    env = QWOPEnv()
    model = DQN.load(MODEL_PATH)

    input('Press Enter to start.')

    time.sleep(1)

    for _ in range(100):

        # Run test
        t = time.time()
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

        print(
            "Test run complete: {:4.1f} in {:4.1f} seconds. Velocity {:2.2f}".format(
                env.previous_score,
                time.time() - t,
                env.previous_score / (time.time() - t),
            )
        )

        time.sleep(1)

        # Admire the finish
        if env.previous_score >= 100:
            time.sleep(5)

    input('Press Enter to exit.')


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
        recorder.generate_obs(env, RECORD_PATH, N_EPISODES)

    if imitate:
        model = get_model(MODEL_PATH)
        imitation_learning.imitate(model, RECORD_PATH, MODEL_PATH, N_EPOCHS)

    if not (test or train or record or imitate):
        with click.Context(main) as ctx:
            click.echo(main.get_help(ctx))


if __name__ == '__main__':
    main()
