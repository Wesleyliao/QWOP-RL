from selenium import webdriver
import time

PORT = 8000

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

    def reset(self):
        
        # Send 'R' key press to restart game
        state = None
        return state
    
    def step(self):
        state, reward, done, info = None, None, None, None
        return state, reward, done, info
    
        
if __name__ == '__main__':
    env = QWOPEnv()
    env.reset()
    print(f'Possible actions: \n{ACTIONS}')
    time.sleep(100000)