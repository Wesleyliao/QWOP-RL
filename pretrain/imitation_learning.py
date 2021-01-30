from stable_baselines.gail import ExpertDataset


def imitate(model, expert_path, model_path, n_epochs=1000):

    dataset = ExpertDataset(expert_path=expert_path + '.npz', batch_size=128)

    model.pretrain(dataset, n_epochs=n_epochs)

    model.save(model_path)
