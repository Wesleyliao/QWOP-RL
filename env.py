from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys 
import time

PORT = 8000
PRESS_DURATION = 0.5
ACTIONS = {
    0: 'qw', 1: 'qo', 2: 'qp', 3: 'q', 4: 'wo', 
    5: 'wp', 6: 'w', 7: 'op', 8: 'o', 9: 'p', 10:''
}

        
class QWOPEnv:
    
    def __init__(self):
        
        # Open browser and go to QWOP page
        self.driver = webdriver.Chrome()
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')
        
        # Wait a bit and then start game
        time.sleep(1)
        self.driver.find_element_by_xpath("//body").click()
        
        # Define action chains
        self.actions = ActionChains(self.driver)

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name}')
    
    def _get_state_(self):
        
        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')
        
        print(game_state)
        print(body_state)
        
        return None
        
        
    def send_keys(self, keys):
        
        if len(keys) > 1:
            keys = list(keys)
        
        self.actions.key_down(*keys)
        self.actions.perform()
        self.actions = ActionChains(self.driver)
        time.sleep(PRESS_DURATION)
        self.actions.key_up(*keys)
        self.actions.perform()
        self.actions = ActionChains(self.driver)
        
    def reset(self):
        
        # Send 'R' key press to restart game
        # self.driver.execute_script(produce_sendkey('r'))
        self.send_keys('r')
        self.send_keys(Keys.SPACE)
        state = False
        return state
    
    def step(self):
        state, reward, done, info = None, None, None, None
        return state, reward, done, info
    
        
if __name__ == '__main__':
    env = QWOPEnv()
    while True:
        env._get_state_()
        time.sleep(3)