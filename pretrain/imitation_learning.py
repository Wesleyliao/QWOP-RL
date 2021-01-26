import os

from stable_baselines import ACER
from stable_baselines.gail import ExpertDataset


def imitate(model, expert_path, model_path):

    if os.path.isfile(model_path + '.zip'):
        print('Loading existing model', model_path)
        model = ACER.load(model_path)

    dataset = ExpertDataset(expert_path=expert_path + '.npz', batch_size=128)

    model.pretrain(dataset, n_epochs=1000)

    model.save(model_path)
