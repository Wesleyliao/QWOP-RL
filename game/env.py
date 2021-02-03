import time

import gym
import numpy as np
from gym import spaces
from pynput.keyboard import Controller
from pynput.keyboard import Key
from selenium import webdriver
from stable_baselines.common.env_checker import check_env

PORT = 8000
PRESS_DURATION = 0.2
STATE_SPACE_N = 71
ACTIONS = {
    0: 'qw',
    1: 'qo',
    2: 'qp',
    3: 'q',
    4: 'wo',
    5: 'wp',
    6: 'w',
    7: 'op',
    8: 'o',
    9: 'p',
    10: '',
}


class QWOPEnv(gym.Env):

    meta_data = {'render.modes': ['human']}
    pressed_keys = set()

    def __init__(self):

        # Open AI gym specifications
        super(QWOPEnv, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[STATE_SPACE_N], dtype=np.float32
        )
        self.num_envs = 1

        # QWOP specific stuff
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.evoke_actions = True

        # Open browser and go to QWOP page
        self.driver = webdriver.Chrome()
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')

        # Wait a bit and then start game
        time.sleep(2)
        self.driver.find_element_by_xpath("//body").click()

        self.keyboard = Controller()

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')

    def _get_state_(self):

        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Get done
        if game_state['gameEnded'] + game_state['gameOver'] > 0:
            self.gameover = done = True
        else:
            self.gameover = done = False

        # Get body state
        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)

        # Get reward
        # if done and game_state['score'] > 100:
        #     reward = game_state['score'] / game_state['scoreTime'] * 1000
        # else:
        #     reward = game_state['score'] - self.previous_score

        reward = max(game_state['score'] - self.previous_score, 0)

        # Update previous scores
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']

        return state, reward, done, {}

    def _release_all_keys_(self):

        for char in self.pressed_keys:
            self.keyboard.release(char)

        self.pressed_keys.clear()

    def send_keys(self, keys):

        # Release all keys
        self._release_all_keys_()

        # Hold down current key
        for char in keys:
            self.keyboard.press(char)
            self.pressed_keys.add(char)

        time.sleep(PRESS_DURATION)

    def reset(self):

        # Send 'R' key press to restart game
        self.send_keys(['r', Key.space])
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self._release_all_keys_()

        return self._get_state_()[0]

    def step(self, action_id):

        # send action
        keys = ACTIONS[action_id]

        if self.evoke_actions:
            self.send_keys(keys)
        else:
            time.sleep(PRESS_DURATION)

        return self._get_state_()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    env = QWOPEnv()
    check_env(env)
    while True:
        if env.gameover:
            env.reset()
        else:
            env.step(env.action_space.sample())
