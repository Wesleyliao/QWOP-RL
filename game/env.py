import time

import gym
import numpy as np
from gym import spaces
from pynput.keyboard import Controller
from pynput.keyboard import Key
from selenium import webdriver
from stable_baselines.common.env_checker import check_env

PORT = 8000
PRESS_DURATION = 0.1
MAX_EPISODE_DURATION_SECS = 120
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
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.evoke_actions = True

        # Open browser and go to QWOP page
        self.driver = webdriver.Chrome()
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')

        # Wait a bit and then start game
        time.sleep(2)
        self.driver.find_element_by_xpath("//body").click()

        self.keyboard = Controller()
        self.last_press_time = time.time()

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')

    def _get_state_(self):

        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Get done
        if (
            (game_state['gameEnded'] > 0)
            or (game_state['gameOver'] > 0)
            or (game_state['scoreTime'] > MAX_EPISODE_DURATION_SECS)
        ):
            self.gameover = done = True
        else:
            self.gameover = done = False

        # Get reward
        torso_x = body_state['torso']['position_x']
        torso_y = body_state['torso']['position_y']

        # Reward for moving forward
        reward1 = max(torso_x - self.previous_torso_x, 0)

        # Penalize for low torso
        if torso_y > 0:
            reward2 = -torso_y / 5
        else:
            reward2 = 0

        # Penalize for torso vertical velocity
        reward3 = -abs(torso_y - self.previous_torso_y) / 4

        # Penalize for bending knees too much
        if (
            body_state['joints']['leftKnee'] < -0.9
            or body_state['joints']['rightKnee'] < -0.9
        ):
            reward4 = (
                min(body_state['joints']['leftKnee'], body_state['joints']['rightKnee'])
                / 6
            )
        else:
            reward4 = 0

        # Combine rewards
        reward = reward1 + reward2 + reward3 + reward4

        # print(
        #     'Rewards: {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #         reward1, reward2, reward3, reward4, reward
        #     )
        # )

        # Update previous scores
        self.previous_torso_x = torso_x
        self.previous_torso_y = torso_y
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']

        # Normalize torso_x
        for part, values in body_state.items():
            if 'position_x' in values:
                values['position_x'] -= torso_x

        # print('Positions: {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #     body_state['torso']['position_x'],
        #     body_state['leftThigh']['position_x'],
        #     body_state['rightCalf']['position_x']
        # ))

        # print('Knee angles: {:3.2f}, {:3.2f}'.format(
        #     body_state['joints']['leftKnee'],
        #     body_state['joints']['rightKnee']
        # ))

        # Convert body state
        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)

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

        # print('pressed for', time.time() - self.last_press_time)
        # self.last_press_time = time.time()
        time.sleep(PRESS_DURATION)

    def reset(self):

        # Send 'R' key press to restart game
        self.send_keys(['r', Key.space])
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
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
