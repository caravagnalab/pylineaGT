from collections import defaultdict
from pyclbr import Function
from tokenize import Double
from typing import Dict
from xmlrpc.client import Boolean
import pyro
import pyro.distributions as distr
import torch
import numpy as np
import sklearn.metrics

from pyro.infer import SVI
from torch.distributions import constraints
from sklearn.cluster import KMeans
from tqdm import trange

class BinMixtureModel():
    def __init__(self):
        pass
