from collections import defaultdict
from pyclbr import Function
from tokenize import Double
from typing import Dict
from xmlrpc.client import Boolean
import pyro
import pyro.distributions as distr
import torch
import numpy as np
import random
import sklearn.metrics

from pyro.infer import SVI, TraceEnum_ELBO, autoguide, config_enumerate
from torch.distributions import constraints
from sklearn.cluster import KMeans
from tqdm import trange


class TSSB():
    def __init__(self):
        raise NotImplementedError

    def tssb_mix_weights(self, nu):
        '''
        `ni`, `psi`, `phi` are three tensors of shape equal to the number of nodes
        '''
        raise NotImplementedError
        # beta1m_cumprod = (1 - beta).cumprod(-1)
        # return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    
    def model(self):
        
        raise NotImplementedError