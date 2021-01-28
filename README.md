# QWOP-RL

## Description

Training a reinforcement learning agent to play QWOP.

## Instructions

Install Python 3.7

```
conda create -n py37 python=3.7
conda activate py37
```

Install dependencies

```
pip install -r requirements.txt
```

Install [Chromedriver](https://sites.google.com/a/chromium.org/chromedriver/home) for
interacting with Chrome through Selenium. Then, start HTTP server for the game

```
python host_game.py
```

Train agent (optional)

```
python main.py --train
```

Test agent

```
python main.py --test
```

To save or use new models, the constant `MODEL_PATH` in `main.py`.

## Acknowledgements

- The original QWOP game at http://www.foddy.net/Athletics.html
- Lachlan for his help in getting QWOP to run offline.
  https://github.com/etopiei/QWOP-Bot
- Keyboard input inspiration from https://github.com/juanto121/qwop-ai
