import torch


def mean_squared_error(prediction, target):
    return torch.mean((prediction - target) ** 2)


def root_mean_squared_error(prediction, target):
    return torch.sqrt(mean_squared_error(prediction, target))


def normalized_rmse(prediction, target, eps=1e-6):
    rmse = root_mean_squared_error(prediction, target)
    scale = torch.sqrt(torch.mean(target**2)) + eps
    return rmse / scale


def compute_metrics(prediction, target):
    return {
        "mse": mean_squared_error(prediction, target).item(),
        "rmse": root_mean_squared_error(prediction, target).item(),
        "nrmse": normalized_rmse(prediction, target).item(),
    }
