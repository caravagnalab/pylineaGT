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


class BayesianRegressionLinear():
    def __init__(self, x, y):
        # x are the timepoints
        # y are the n of cells per timepoint, with a dimension per lineage
        self.x = x  # timepoints
        self.y = self.check_zero(y)  # the n of cells
        
        self.N = self.y.shape[0]
        self.L = self.y.shape[1]  # n of lineages

        self.init_params = {"is_computed":False}


    def check_zero(self, x):
        x = x.float()
        x[x==0] = 1e-20
        return x


    def model(self):
        with pyro.plate("lineages", self.L):
            rate = pyro.sample("rate", distr.Normal(0., .5))
            # rate = pyro.sample("rate", distr.Normal(0.0, 1.0))
        # print(rate)

        # the intercept is ln(1) = 0
        # sigma prior for the random error
        sigma = pyro.sample("sigma", distr.HalfNormal(1.0))
        mean = self.x * rate  # multiply each n_cell by the sampled rate

        for ll in pyro.plate("lineages2", self.L): ## pyro.plate("lineages", self.L):
            with pyro.plate(f"data_{ll}", self.N):
                # Condition the expected mean on the observed target y
                obs = pyro.sample(f"obs_{ll}", distr.Normal(mean[:,ll], sigma), 
                    obs=torch.log(self.y[:,ll]))


    def guide(self):
        return pyro.infer.autoguide.AutoDelta(self.model)
        # params = self.initialize_params()

        # rate_param = pyro.param("rate_param", lambda: params["rate"])
        # with pyro.plate("lineages", self.L):
        #     rate = pyro.sample("rate", distr.Delta(rate_param))

        # sigma = pyro.sample("sigma", distr.HalfNormal(1.))


    def initialize_params(self):
        if not self.init_params["is_computed"]:
            rate = torch.ones((self.L))
            # for ll in range(self.L):
            #     rate[ll] = distr.Normal(0.,1.).sample()

            self.init_params["rate"] = rate
            self.init_params["is_computed"] = True
        
        return self.init_params


    def train(self, steps=100, optim=pyro.optim.Adam({"lr": 0.03})):
        pyro.clear_param_store()
        self._global_guide = self.guide()
        svi = SVI(self.model, self._global_guide, optim, Trace_ELBO())

        # Train (i.e. do ELBO optimization) for num_steps iterations
        losses = []
        t = trange(steps, desc='Bar desc', leave=True)
        for step in range(steps):
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


class BayesianRegressionLogistic():
    def __init__(self, x, y):
        # x are the timepoints
        # y are the n of cells per timepoint, with a dimension per lineage
        self.x = x  # timepoints
        self.y = y  # the n of cells
        
        self.N = self.y.shape[0]
        self.L = self.y.shape[1]  # n of lineages

    # def model(self):
    #     w = pyro.sample("w", distr.MultivariateNormal(torch.ones(self.dataset.shape[1]), 
    #         torch.eye(self.dataset.shape[1]))) 
    #     with pyro.plate("data", self.dataset.shape[0]): 
    #         sigmoid = torch.sigmoid(torch.matmul(torch.tensor(self.dataset).double(), 
    #             torch.tensor(w).double())) 
    #         x = pyro.sample("x", distr.Bernoulli(sigmoid), obs=self.dataset) 
    
    # def guide(self):
    #     variance_q = pyro.param("variance", torch.eye(self.dataset.shape[1]), 
    #         torch.distributions.constraints.positive)
    # 
    #     mu_q = pyro.param("mu", torch.zeros(self.dataset.shape[1]))
    #     w = pyro.sample("w", distr.MultivariateNormal(mu_q, variance_q))



# class Regression():
#     def __init__(self, time, data, log_exp="exp"):
#         '''
#         - `time` -> a list/tensor/array with the temporal coordinates.
#         - `data` -> a matrix/tensor/array with shape `(L,T,K)`, where `L` is the 
#         number of lineages, `T` is the number of timepoints and `K` is the number of
#         clones for each lineage.
#         '''
#         self.dataset = torch.tensor(data)
#         self.time = torch.tensor(time).expand_as(self.dataset)

#         self.L, self.T, self.K = self.dataset.shape
#         self._initialize_attributes()

#         self.input_features = 1
#         self.output_features = self.K  # the number of clusters

#         self.log_exp = log_exp


#     def _initialize_attributes(self):
#         self.params = {"T":self.T, "K":self.K, "L":self.L}


#     def model(self):
#         if self.log_exp == "exp":
#             return BayesianRegressionLinear(self.input_features, self.output_features)
        
#         if self.log_exp == "log":
#             return BayesianRegressionLogistic(self.input_features, self.output_features)


#     def guide(self, model=None):
#         if model is None:
#             model = self.model()
#         return AutoDelta(model)


#     def _train(self, steps=1000, lr=0.005, loss_fn=Trace_ELBO()):
#         pyro.clear_param_store() 
#         optimizer = Adam({"lr": lr})

#         model = self.model()
#         guide = self.guide(model)

#         svi = SVI(model, guide, optimizer, loss=loss_fn)

#         losses = []
#         for step in range(steps):
#             # calculate the loss and take a gradient step
#             loss = svi.step(self.dataset[:,0].unsqueeze(1), self.dataset[:,1].unsqueeze(1))
#             losses.append(loss)

#             if step % 50 == 0:
#                 print("[iteration %04d] loss: %.4f" % (step + 1, loss / float(self.dataset.shape[0])))
        
#         return losses