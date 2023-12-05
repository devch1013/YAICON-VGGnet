import torch
import torch.optim as optim

def get_optimizer(model, cfg):
    """
    Return torch optimizer

    Args:
        model: Model you want to train
        cfg: Dictionary of optimizer configuration

    Returns:
        optimizer
    """
    optim_name = cfg["name"].lower()
    learning_rate = cfg["learning-rate"]
    args = cfg["args"]
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, **args)
    else:
        raise NotImplementedError
    return optimizer


def get_criterion(cfg):
    """
    Return torch criterion

    Args:
        cfg: Dictionary of criterion configuration

    Returns:
        criterion
    """
    criterion_name = cfg["name"].lower()
    if criterion_name == "crossentropyloss":
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_name == "mseloss":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "msecosembloss":
        criterion = cosembloss
    else:
        raise NotImplementedError
    return criterion

def cosembloss(output, target):
    mse = torch.nn.MSELoss()
    cosemb = torch.nn.CosineEmbeddingLoss()
    y = torch.tensor([1])
    return mse(output, target) + cosemb(output, target,y)
    

def get_scheduler(optimizer, cfg):
    """
    get ["lr_scheduler"] cfg dictionary
    """
    scheduler_name = cfg["name"].lower()
    args = cfg["args"]
    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", **args)
    elif scheduler_name == "multisteplr":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, **args)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler