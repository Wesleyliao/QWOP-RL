from stable_baselines.gail import ExpertDataset


def imitate(model, expert_path, model_path, learning_rate, n_epochs=1000):

    dataset = ExpertDataset(expert_path=expert_path + '.npz', batch_size=128)

    model.pretrain(dataset, n_epochs=n_epochs, learning_rate=learning_rate)

    model.save(model_path)
