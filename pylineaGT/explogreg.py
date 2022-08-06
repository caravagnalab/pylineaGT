from copy import copy, deepcopy
from random import random
import pyro
import pyro.distributions as distr
import torch
import numpy as np

from pyro.infer import SVI, Trace_ELBO
from tqdm import trange


class Regression():
    def __init__(self, x, y):
        '''
        - `x` -> a tensor of integer values corresponding to time coordinates.
        - `y` -> a tensor with shape `[N,L]`, being `L` the number of lineages.
        '''
        self.N = x.shape[0]  # n of timepoints
        self.x = x.reshape((self.N,1)) if len(x.shape) == 2 else x  # timepoints
        self.y = self.check_zero(y)  # to avoid log of 0, change 0s to 1e-20
        
        self.L = self.y.shape[1]  # n of lineages
        self.compute_initial_t()


    def check_zero(self, x):
        x = x.float()
        x[x==0] = 1e-20
        return x


    def compute_initial_t(self):
        '''
        Function to compute the times `t_i` at which the subclone appears in each lineage.
        '''
        t = torch.zeros(self.L)
        # for each lineage, I store the index of the first value greater than 0
        for ll in range(self.L):
            tmp = self.y[:,ll]
            try: t[ll] = self.x[tmp>11e-20][0]
            except: t[ll] = torch.max(self.x)
            # except: t[ll] = torch.tensor(0.)
        
        self.init_time = t.int()
        
        self._estimate_t1 = True
        if self.init_time.equal(torch.zeros_like(self.init_time)):  
            # if a t0 we observe the pop in all lineages
            # we are in the case of the WT clonal population
            self._estimate_t1 = False


    def model_logistic(self):
        '''
        `y(x) = K / [ 1 + (K-1)e^(-wx) ] = K / [ 1 + e^-( -ln(K-1) + wx ) ]`

        Using a Bernoulli likelihood
        - `y(x) = K * logit( -ln(K-1) + wx )`
        - `y(x) / K = logit( -ln(K-1) + wx )`

        Therefore: compute the argument of the logit and pass it as `logits` parameter of 
        the Bernoulli likelihood, with observed values `y/K`.
        '''
        ## mm is the maximum observed population value per lineage
        unif_low = torch.max(self.y, dim=0).values.ceil()
        unif_high = torch.max(unif_low*2, unif_low+1)
        t1 = self.init_time

        with pyro.plate("lineages", self.L):
            fitn = pyro.sample("fitness", distr.Normal(0., .5))

            ## TODO improve the limits of the Uniform ?
            carrying_capacity = pyro.sample("carr_capac", distr.Uniform(low=unif_low, high=unif_high))

            # estimate the t0 for the subclones
            # t0 is the value s.t. K * logit( -ln(K-1) + wt0 ) = 1
            # t0 is the value s.t. e^(rate*t0) = 1 -> t0 = ln(1) / rate AND t0 > 0
            if self._estimate_t1:
                t1 = pyro.sample("init_time", distr.Uniform(1, self.init_time.float()))

        with pyro.plate("obs_sigma", self.N):
            sigma = pyro.sample("sigma", distr.HalfNormal(1))

        rate = fitn if self.p_rate is None else self.p_rate*(1+fitn)
        logits = (self.x.expand(self.N, self.L) - t1).clamp(0) * rate - \
            torch.log(carrying_capacity -1)

        for ll in pyro.plate("lineages2", self.L):
            with pyro.plate(f"data_{ll}", self.N):
                obs = pyro.sample(f"obs_{ll}", distr.Bernoulli(logits=logits[:,ll] + sigma, validate_args=False), 
                    obs=self.y[:,ll] / carrying_capacity[ll])


    def model_exp(self):
        t1 = self.init_time

        with pyro.plate("lineages", self.L):
            fitn = pyro.sample("fitness", distr.Normal(0., .5))

            # estimate the t0 for the subclones
            # t0 is the value s.t. e^(rate*t0) = 1 -> t0 = ln(1) / rate AND t0 > 0
            if self._estimate_t1:
                t1 = pyro.sample("init_time", distr.Uniform(1, self.init_time.float()))

        rate = fitn if self.p_rate is None else self.p_rate*(1+fitn)

        with pyro.plate("obs_sigma", self.N):
            sigma = pyro.sample("sigma", distr.HalfNormal(1))

        # for each lineage, the pop grows as r*(t-t1), 
        # where t1 is the time the population is first observed
        mean = (self.x.expand(self.N, self.L) - t1).clamp(0) * rate

        for ll in pyro.plate("lineages2", self.L):
            with pyro.plate(f"data_{ll}", self.N):
                obs = pyro.sample(f"obs_{ll}", distr.Normal(mean[:,ll], sigma), 
                    obs=torch.log(self.y[:,ll]))


    def guide(self):
        return pyro.infer.autoguide.AutoDelta(self._global_model) 


    def _set_regr_type(self, regr):
        self.exp = True if "exp" in regr else False
        self.log = True if "log" in regr else False


    def train(self, steps=500, optim=pyro.optim.Adam, lr=0.01, loss_fn=pyro.infer.Trace_ELBO(), \
            regr="exp", p_rate=None, min_steps=50, p=.5, random_state=25):
        
        pyro.enable_validation(True)
        # to clear the param store
        pyro.get_param_store().__init__()

        if random_state is not None:
            pyro.set_rng_seed(random_state)

        self._set_regr_type(regr)
        self.p_rate = p_rate
        
        self._settings = {"optim":optim({"lr": lr}), "lr":lr, "loss":loss_fn}
        self._global_model = self.model_exp if self.exp else self.model_logistic if self.log else None
        self._global_guide = self.guide()
        self.svi = SVI(self._global_model, self._global_guide, self._settings["optim"], Trace_ELBO())

        conv = 0
        losses = []
        t = trange(steps, desc='Bar desc', leave=True)
        for step in t:
            # print(step)
            loss = self.svi.step() / self.N
            losses.append(loss)

            params_step = self._global_guide()
            if step == 1:
                rate_conv = [0, params_step["fitness"].clone().detach()]
                carrying_conv = None if not self.log else [0, params_step["carr_capac"].clone().detach()]
                t1_conv = None if not self._estimate_t1 else [0, params_step["init_time"].clone().detach()]

            if step >= min_steps:
                rate_conv[0], rate_conv[1] = rate_conv[1], params_step["fitness"].clone().detach()
                if self.log:
                    carrying_conv[0], carrying_conv[1] = carrying_conv[1], params_step["carr_capac"].clone().detach()
                if self._estimate_t1:
                    t1_conv[0], t1_conv[1] = t1_conv[1], params_step["init_time"].clone().detach()
                
                conv = self._convergence(rate_conv, carrying_conv, t1_conv, conv, p=p)
                if conv == 10:
                    t.set_description("ELBO %f" % loss)
                    t.reset(total=step)
                    break
            
            t.set_description("ELBO %f" % loss)
            t.refresh()
        return losses


    def _convergence(self, rate_conv, carrying_conv, t1_conv, conv, p):
        if self._check_convergence(rate_conv, p) and \
                self._check_convergence(carrying_conv, p) and \
                self._check_convergence(t1_conv, p):
            return conv + 1
        return 0


    def _check_convergence(self, par, p=.01):
        '''
        - `par` -> list of 2 elements given the previous (at index 0) and current (at index 1) 
        estimated values for a parameter.
        - `p` -> numeric value in `[0,1]`, corresponding to a percentage of the learning rate used for convergence.
        The function returns `True` if more than 95% of the values changed less than `perc*100`% 
        in the current step with respect to the previous one.
        '''
        if par is None:
            return True
        par = self._normalize(par)
        eps = p * self._settings["lr"]
        n = 0
        for ll in range(par[0].shape[0]):
            perc = eps * par[0][ll] # torch.max(torch.tensor(1), par[0][k,t])
            if torch.absolute(par[0][ll] - par[1][ll]) <= perc:
                n += 1

        return n == self.L


    def _normalize(self, par):
        # zi = (xi - min(x)) / (max(x) - min(x))
        norm = list()
        for p in par:
            if torch.min(p) == torch.max(p):
                norm.append(p)
            norm.append(( p - torch.min(p) ) / ( torch.max(p) - torch.min(p) ) * 100)
        return norm


    def get_learned_params(self, return_numpy=True):
        params = {}
        self._global_guide.requires_grad_(False)
        for name, value in self._global_guide().items():
            ## NOTE do NOT first detach() and then clone() because it will detach the accumulated gradient
            params[name] = value.clone().detach()
            if return_numpy:
                params[name] = params[name].numpy()

        if return_numpy:
            params["init_time"] = params.get("init_time", self.init_time.clone().detach().numpy())
            params["parent_rate"] = self.p_rate.clone().detach().numpy() if self.p_rate is not None else None
        else:
            params["init_time"] = params.get("init_time", self.init_time.clone().detach())
            params["parent_rate"] = self.p_rate.clone().detach() if self.p_rate is not None else None

        return params


    def compute_log_likelihood(self):
        if self.exp:
            return self._compute_exp_ll()

        if self.log:
            return self._compute_log_ll()


    def _compute_log_ll(self):
        params = self.get_learned_params(return_numpy=False)

        rate = params["fitness"] if self.p_rate is None else self.p_rate*(1+params["fitness"])
        logits = (self.x.expand(self.N, self.L) - params["init_time"]).clamp(0) * rate - \
            torch.log(params["carr_capac"] -1)

        sigma = params["sigma"].unsqueeze(1)

        return torch.sum(distr.Bernoulli(logits=logits + sigma, \
            validate_args=False).log_prob(self.y / params["carr_capac"]), dim=0).detach().numpy()


    def _compute_exp_ll(self):
        params = self.get_learned_params(return_numpy=False)

        rate = params["fitness"] if self.p_rate is None else self.p_rate*(1+params["fitness"])
        mean = (self.x.expand(self.N, self.L) - params["init_time"]).clamp(0) * rate
        sigma = params["sigma"].unsqueeze(1)

        return torch.sum(distr.Normal(mean, sigma).log_prob(torch.log(self.y)), dim=0).detach().numpy()
