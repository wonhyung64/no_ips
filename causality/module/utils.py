import torch
import random
import numpy as np


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cpu"

    return device


def estimate_ips_bayes(x, y, y_ips=None):
    if y_ips is None:
        one_over_zl = np.ones(len(y))
    else:
        prob_y1 = y_ips.sum() / len(y_ips)
        prob_y0 = 1 - prob_y1
        prob_o1 = len(x) / (x[:,0].max() * x[:,1].max())
        prob_y1_given_o1 = y.sum() / len(y)
        prob_y0_given_o1 = 1 - prob_y1_given_o1

        propensity = np.zeros(len(y))

        propensity[y == 0] = (prob_y0_given_o1 * prob_o1) / prob_y0
        propensity[y == 1] = (prob_y1_given_o1 * prob_o1) / prob_y1
        one_over_zl = 1 / propensity

    one_over_zl = torch.Tensor(one_over_zl)

    return one_over_zl


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
