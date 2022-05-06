import pyro
import pyro.distributions as distr
import torch
import numpy as np
import re

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam


class ExpRegression():
    def __init__(self, data, mutations=[], columns=[]):
        '''
        - `data` -> a matrix/tensor with three columns: `time`, `y` and `labels`, where `time` 
        contains numeric values representing the time, `y` represents the VAFs and `labels` 
        annotates each observation to the respective cluster.
        - `columns` -> name of the dimensions considered in the inference (columns of the dataset).
        - `mutations` -> values of the mutations, one per row.
        '''
        # input dataset with columns:
        # mutation, vaf_early/mid/late, lineage
        self._set_dataset(data)
        self.mutations = np.array(mutations)
        self.dimensions = columns # `columns` will be a list of type ["early.MNC", "mid.MNC", "early.Myeloid", "mid.Myeloid"]

        self._N, self._T = self.dataset.shape[0], self.dataset.shape[1]
        self._initialize_attributes()


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


    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def _compute_independent_reg(self):
        '''
        Function to compute the independent regressions, one for each cluster, to estimate the
        values `Y_0` and `w_0`
        '''
        raise NotImplementedError
