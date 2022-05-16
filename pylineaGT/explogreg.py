import pyro
import pyro.distributions as distr
import torch
import numpy as np

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroSample


class BayesianRegressionLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[torch.nn.Linear](in_features, out_features)
        
        self.linear.weight = PyroSample(distr.Uniform(-.1,.1).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(distr.Normal(0.,1.).expand([out_features]).to_event(1))

    def forward(self, x, y):
        sigma = pyro.sample("sigma", distr.HalfNormal(0.2))
        mean = self.linear(x)
        
        for l in pyro.plate("lineages", x.shape[0]):
            for t in pyro.plate(f"data_{l}", x.shape[1]):
                for k in pyro.plate(f"clusters_{l}.{t}", x.shape[2]):
                    obs = pyro.sample(f"obs_{l}.{t}.{k}", distr.Normal(mean[l,t,k], sigma), obs=torch.log(y[l,t,k]))
        return mean


class BayesianRegressionLogistic(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()

    def forward(self, x, y):
        raise NotImplementedError


class Regression():
    def __init__(self, time, data, log_exp="exp"):
        '''
        - `time` -> a list/tensor/array with the temporal coordinates.
        - `data` -> a matrix/tensor/array with shape `(L,T,K)`, where `L` is the 
        number of lineages, `T` is the number of timepoints and `K` is the number of
        clones for each lineage.
        '''
        self.dataset = torch.tensor(data)
        self.time = torch.tensor(time).expand_as(self.dataset)

        self.L, self.T, self.K = self.dataset.shape
        self._initialize_attributes()

        self.input_features = 1
        self.output_features = self.K  # the number of clusters

        self.log_exp = log_exp


    def _initialize_attributes(self):
        self.params = {"T":self.T, "K":self.K, "L":self.L}


    def model(self):
        if self.log_exp == "exp":
            return BayesianRegressionLinear(self.input_features, self.output_features)
        
        if self.log_exp == "log":
            return BayesianRegressionLogistic(self.input_features, self.output_features)


    def guide(self, model=None):
        if model is None:
            model = self.model()
        return AutoDelta(model)


    def _train(self, steps=1000, lr=0.005, loss_fn=Trace_ELBO()):
        pyro.clear_param_store() 
        optimizer = Adam({"lr": lr})

        model = self.model()
        guide = self.guide(model)

        svi = SVI(model, guide, optimizer, loss=loss_fn)

        losses = []
        for step in range(steps):
            # calculate the loss and take a gradient step
            loss = svi.step(self.dataset[:,0].unsqueeze(1), self.dataset[:,1].unsqueeze(1))
            losses.append(loss)

            if step % 50 == 0:
                print("[iteration %04d] loss: %.4f" % (step + 1, loss / float(self.dataset.shape[0])))
        
        return losses