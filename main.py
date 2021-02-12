import os
import time

import click
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import SubprocVecEnv

from game.env import QWOPEnv
from pretrain import imitation_learning
from pretrain import recorder

# Training parameters
MODEL_NAME = 'PPO2_imitateACER'
LEARNING_RATE = 0.00025 * (1 / 30)
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


def get_new_model():

    # Initialize env and model
    env = QWOPEnv()
    model = PPO2(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_PATH,
    )

    return model


def get_existing_model(model_path):

    print('--- Training from existing model', model_path, '---')

    # Load model
    model = PPO2.load(model_path, tensorboard_log=TENSORBOARD_PATH)

    # Set environment
    env = SubprocVecEnv([lambda: QWOPEnv()])
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
    model = PPO2.load(MODEL_PATH)

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
