# QWOP-RL

## Description

Training a reinforcement learning agent to play QWOP.

## Instructions

Install dependencies

```
pip install pipenv
pipenv install
pipenv shell
```

Install Chromedriver for hosting the game https://sites.google.com/a/chromium.org/chromedriver/home. Then, start HTTP server for the game

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

You can change the constant `MODEL_PATH` in `main.py` so save new models and use them
for testing.

## Acknowledgements

- The original QWOP game at http://www.foddy.net/Athletics.html
- Huge shoutouts to Lachlan for his help in getting QWOP to run offline.
  https://github.com/etopiei/QWOP-Bot
- Key input inspiration from https://github.com/juanto121/qwop-ai
