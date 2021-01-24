from selenium import webdriver
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.keys import Keys 
from pynput.keyboard import Key, Controller
import numpy as np
import time

PORT = 8000
PRESS_DURATION = 0.3
ACTIONS = {
    0: 'qw', 1: 'qo', 2: 'qp', 3: 'q', 4: 'wo', 
    5: 'wp', 6: 'w', 7: 'op', 8: 'o', 9: 'p', 10:''
}

        
class QWOPEnv:
    
    def __init__(self):
        
        self.gameover = False
        
        # Open browser and go to QWOP page
        self.driver = webdriver.Chrome()
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')
        
        # Wait a bit and then start game
        time.sleep(1)
        self.driver.find_element_by_xpath("//body").click()
        
        self.keyboard = Controller()
        
    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')
    
    def _get_state_(self):
        
        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')
        
        if game_state['gameEnded'] + game_state['gameOver'] > 0:
            self.gameover = True
            
        # print(game_state)
        # print(body_state)
        
        return None
        
        
    def send_keys(self, keys):
        
        for char in keys:
            print(f'pressing {char}')
            self.keyboard.press(char)
        
        time.sleep(PRESS_DURATION)
        
        for char in keys:
            self.keyboard.release(char)
        
    def reset(self):
        
        # Send 'R' key press to restart game
        self.send_keys(['r', Key.space])
        self.gameover = False
        state = False
        return state
    
    def step(self):
        state, reward, done, info = None, None, None, None
        return state, reward, done, info
    
        
if __name__ == '__main__':
    env = QWOPEnv()
    actions = list(ACTIONS.values())
    while True:
        env._get_state_()
        if env.gameover:
            env.reset()
        else:
            env.send_keys(np.random.choice(actions))
        time.sleep(.5)