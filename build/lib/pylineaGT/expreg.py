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

from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, autoguide, config_enumerate
from torch.distributions import constraints
from sklearn.cluster import KMeans
from tqdm import trange


class ExpRegression():
    def __init__(self, data, IS=[], columns=[], lineages=[]):
        self._set_dataset(data)
        self.IS = np.array(IS)
        self.dimensions = columns # `columns` will be a list of type ["early.MNC", "mid.MNC", "early.Myeloid", "mid.Myeloid"]
        self.lineages = lineages

        self.K, self._N, self._T = K, self.dataset.shape[0], self.dataset.shape[1]
        self._initialize_attributes()
        self._initialize_sigma_constraint() 


    def _set_dataset(self, data):
        if isinstance(data, torch.Tensor):
            self.dataset = data.int()
        else:
            try:
                try: self.dataset = torch.tensor(data.values)
                except: pass
                try: self.dataset = torch.tensor(data)
                except: pass
            except: pass


    def _initialize_attributes(self):
        self.params = {"N":self._N, "K":self.K, "T":self._T}
        self.init_params = {"N":self._N, "K":self.K, "T":self._T, "is_computed":False,\
            "sigma":None, "mean":None, "weights":None, \
            "clusters":None, "var_constr":None}
        self.hyperparameters = {"mean_scale":self.dataset.float().var(), \
            "mean_loc":self.dataset.float().mean(), "var_scale":100, "eta":1}
        self._autoguide = False
        self._enumer = "parallel"

        if len(self.dimensions) == 0:
            self.dimensions = [str(_) for _ in range(self._T)]
