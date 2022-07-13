import pyro
import pyro.distributions as distr
import torch
import numpy as np
import sys

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample
from tqdm import trange


class Regression():
    def __init__(self, x, y, p_rate=None):
        '''
        - `x` -> a tensor of integer values corresponding to time coordinates.
        - `y` -> a tensor with shape `[N,L]`, being `L` the number of lineages.
        '''
        self.N = x.shape[0]
        self.x = x.reshape((self.N,1)) if len(x.shape) == 2 else x  # timepoints
        self.y = self.check_zero(y)  # the n of cells
        
        self.L = self.y.shape[1]  # n of lineages
        self.t_zeros = self.compute_initial_t()
        self.p_rate = p_rate


    def check_zero(self, x):
        x = x.float()
        x[x==0] = 1e-20
        return x


    def compute_initial_t(self):
        '''
        Function to compute the times `t_i` at which the subclone appears in each lineage.
        '''
        t = torch.zeros(self.L)
        ## for each lineage, I store the index of the first value greater than 0
        for ll in range(self.L):
            tmp = self.y[:,ll]
            t[ll] = self.x[tmp>=1][0]
        
        self.init_t = t.int()


    def model_logistic(self):
        '''
        `y(x) = K / [ 1 + (K-1)e^(-wx) ] = K / [ 1 + e^-( -ln(K-1) + wx ) ]`

        Using a Bernoulli likelihood
        - `y(x) = K * logit( -ln(K-1) + wx )`
        - `y(x) / K = logit( -ln(K-1) + wx )`

        Therefore: compute the argument of the logit and pass it as `logits` parameter of 
        the Bernoulli likelihood, with observed values `y/K`.
        '''
        mm = torch.max(self.y, dim=0).values

        with pyro.plate("lineages", self.L):
            # rate = pyro.sample("rate", distr.Normal(0., 1))
            fitn = pyro.sample("fitness", distr.Normal(0., 1))
            carrying_capacity = pyro.sample("carr_capac", distr.Uniform(low=mm, high=mm*2))

        sigma = pyro.sample("sigma", distr.HalfNormal(.5))
        
        rate = fitn if self.p_rate is None else self.p_rate*(1+fitn)
        # logits = rate * (self.x) - torch.log(carrying_capacity -1) + sigma
        logits = (self.x.expand(self.N, self.L) - self.init_t).clamp(0) * rate - \
            torch.log(carrying_capacity -1) + sigma

        for ll in pyro.plate("lineages2", self.L):
            with pyro.plate(f"data_{ll}", self.N):
                obs = pyro.sample(f"obs_{ll}", distr.Bernoulli(logits=logits[:,ll], validate_args=False), 
                    obs=self.y[:,ll] / carrying_capacity[ll])


    def model_exp(self):
        with pyro.plate("lineages", self.L):
            fitn = pyro.sample("fitness", distr.Normal(0., 1))
            # rate = pyro.sample("rate", distr.Normal(0., 1))

        rate = fitn if self.p_rate is None else self.p_rate*(1+fitn)
        sigma = pyro.sample("sigma", distr.HalfNormal(.5))
        mean = (self.x.expand(self.N, self.L) - self.init_t).clamp(0) * rate

        for ll in pyro.plate("lineages2", self.L):
            with pyro.plate(f"data_{ll}", self.N):
                obs = pyro.sample(f"obs_{ll}", distr.Normal(mean[:,ll], sigma), 
                    obs=torch.log(self.y[:,ll]))


    def guide(self):
        return pyro.infer.autoguide.AutoDelta(self._global_model) 
        # if self.exp:
        #     return pyro.infer.autoguide.AutoDelta(self.model_exp)
        # if self.log:
        #     return pyro.infer.autoguide.AutoDelta(self.model_logistic)

    
    def _set_regr_type(self, regr):
        self.exp = True if "exp" in regr else False
        self.log = True if "log" in regr else False


    def train(self, steps=100, optim=pyro.optim.Adam({"lr": 0.03}), regr="exp"):
        pyro.clear_param_store()

        self._set_regr_type(regr)

        self._global_model = self.model_exp if self.exp else self.model_logistic if self.log else None
        self._global_guide = self.guide()
        svi = SVI(self._global_model, self._global_guide, optim, Trace_ELBO())

        losses = []
        t = trange(steps, desc='Bar desc', leave=True)
        for step in t:
            # print(step)
            loss = svi.step() / self.N
            losses.append(loss)
            
            t.set_description("ELBO %f" % loss)
            t.refresh()
        return losses


    def get_learned_params(self):
        params = {}
        self._global_guide.requires_grad_(False)
        for name, value in pyro.get_param_store().items():
            params[name] = value.detach().clone()
        return params
