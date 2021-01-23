from selenium import webdriver
import time

PORT = 8000

class qwop_env:
    
    def __init__(self):
        
        driver = webdriver.Chrome()
        driver.get(f'localhost:{PORT}/Athletics.html')


if __name__ == '__main__':
    env = qwop_env()
    time.sleep(100000)