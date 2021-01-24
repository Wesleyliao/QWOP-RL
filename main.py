from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from game.env import QWOPEnv

TRAIN_TIME_STEPS = 60000
MODEL_PATH = "models/PPO2_MLP_v1"


def train():
    env = QWOPEnv()
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=TRAIN_TIME_STEPS)
    model.save(MODEL_PATH)


def test():
    env = QWOPEnv()
    model = PPO2.load(MODEL_PATH)
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


if __name__ == '__main__':
    # train()
    test()
