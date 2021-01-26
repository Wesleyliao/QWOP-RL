from stable_baselines.gail import ExpertDataset


def imitate(model, expert_path, model_path):

    dataset = ExpertDataset(expert_path=expert_path, traj_limitation=1, batch_size=128)

    model.pretrain(dataset, n_epochs=1000)

    model.save(model_path)
