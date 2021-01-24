import time

import numpy as np
from pynput.keyboard import Controller
from pynput.keyboard import Key
from selenium import webdriver

PORT = 8000
PRESS_DURATION = 0.3
ACTIONS = {
    0: 'qw', 1: 'qo', 2: 'qp', 3: 'q', 4: 'wo',
    5: 'wp', 6: 'w', 7: 'op', 8: 'o', 9: 'p', 10: ''
}


class QWOPEnv:

    def __init__(self):

        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0

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
            state.append(list(part.values()))
        state = np.array(state).flatten()

        # Get reward
        if not done:
            reward = game_state['score'] - self.previous_score
        else:
            reward = game_state['score']

        # Update previous scores
        self.previous_score = game_state['score']
        self.previous_time = game_state['scoreTime']

        return state, reward, done, None

    def send_keys(self, keys):

        for char in keys:
            self.keyboard.press(char)

        time.sleep(PRESS_DURATION)

        for char in keys:
            self.keyboard.release(char)

    def reset(self):

        # Send 'R' key press to restart game
        self.send_keys(['r', Key.space])
        self.gameover = False

        return self._get_state_()

    def step(self, action_id):

        # send action
        keys = ACTIONS[action_id]
        self.send_keys(keys)

        return self._get_state_()


if __name__ == '__main__':
    env = QWOPEnv()
    time.sleep(.5)
    while True:
        if env.gameover:
            env.reset()
        else:
            env.step(np.random.randint(11))
