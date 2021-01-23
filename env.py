from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
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

    def send_keys(self, keys):
        self.actions.send_keys(keys)
        self.actions.perform()
        
    def reset(self):
        
        # Send 'R' key press to restart game
        # self.driver.execute_script(produce_sendkey('r'))
        self.actions.send_keys('r')
        self.actions.perform()
        state = False
        return state
    
    def step(self):
        state, reward, done, info = None, None, None, None
        return state, reward, done, info
    
        
if __name__ == '__main__':
    env = QWOPEnv()
    while True:
        env.reset()
        time.sleep(1)
    # print(f'Possible actions: \n{ACTIONS}')
    time.sleep(100000)