from cProfile import label
from run import run_inference
import torch
import pyro.distributions as distr
import pickle
import os
import pyro

class Simulate():
    def __init__(self, seed, N=200, T=5, K=15, 
        mean_loc=500, mean_scale=1000, 
        var_loc=118, var_scale=130, min_var=5,
        eta=1, cov_type="full", label=""):

        self.settings = {"N":N, "T":T, "K":K, 
            "mean_loc":torch.tensor(mean_loc).float(), 
            "mean_scale":torch.tensor(mean_scale).float(),
            "var_loc":torch.tensor(var_loc).float(), 
            "var_scale":torch.tensor(var_scale).float(), 
            "min_var":torch.tensor(min_var).float(),
            "eta":torch.tensor(eta).float(),
            "slope":torch.tensor(0.15914), "intercept":torch.tensor(23.70988),
            # "slope":0.09804862, "intercept":22.09327233,
            "seed":seed}

        pyro.set_rng_seed(seed)

        self.cov_type = cov_type
        self.label = label
        # self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), str(label)])

        self.set_sigma_constraints()


    def set_sigma_constraints(self):
        slope = self.settings["slope"]
        intercept = self.settings["intercept"]

        T = self.settings["T"]
        slope_tns = torch.repeat_interleave(slope, T)
        intercept_tns = torch.repeat_interleave(intercept, T)
        self.lm = {"slope":slope_tns, "intercept":intercept_tns}
        self.settings["slope"] = slope_tns
        self.settings["intercept"] = intercept_tns


    def generate_dataset(self):
        K = self.settings["K"]
        T = self.settings["T"]
        N = self.settings["N"]
        mean_scale = self.settings["mean_scale"]
        mean_loc = self.settings["mean_loc"]
        var_loc = self.settings["var_loc"]
        var_scale = self.settings["var_scale"]
        min_var = self.settings["min_var"]
        eta = self.settings["eta"]

        weights = distr.Dirichlet(torch.ones(K)).sample()
        
        z = torch.zeros((N,), dtype=torch.long)
        x = torch.zeros((N,T))
        for n in range(N):
            z[n] = distr.Categorical(weights).sample()

        self.settings["K"] = K = len(z.unique())
        labels = z.unique()

        weights = weights[labels]
        weights = weights / torch.sum(weights)        
        
        tmp = {int(k):v for v,k in enumerate(labels)}  # k is the old value, v is the new value
        z = torch.tensor([tmp[int(z[i])] for i in range(len(z))])

        mean = torch.zeros(K,T)
        sigma_vector = torch.zeros(K,T)
        var_constr = torch.zeros(K,T)
        
        for k in range(K):  # for each cluster
            mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))
            sigma_vector[k,:] = distr.Normal(var_loc, var_scale).sample(sample_shape=(T,))

            # check for negative values
            while torch.any(mean[k,:] < 0):
                mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))

            var_constr[k,:] = mean[k,:] * self.lm["slope"] + self.lm["intercept"]

            # check for negative values
            while torch.any(sigma_vector[k,:] < min_var) or torch.any(sigma_vector[k,:] > var_constr[k,:]):
                sigma_vector[k,:] = distr.Normal(var_loc, var_scale).sample(sample_shape=(T,))

        if self.cov_type == "diag" or T==1:
            sigma_chol = torch.eye(T) * 1.
        
        if self.cov_type == "full" and T>1:
            sigma_chol = distr.LKJCholesky(T, eta).sample(sample_shape=(K,))

        Sigma = self._compute_Sigma(sigma_chol, sigma_vector, K)
        for n in range(N):
            x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
            while torch.any(x[n,:] < 0):
                x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()

        self.dataset = x
        self.params = {"weights":weights, "mean":mean, "sigma":sigma_vector, "var_constr":var_constr, "z":z}

        self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), str(self.label)])


    def _compute_Sigma(self, sigma_chol, sigma_vector, K):
        '''
        Function to compute the sigma_tril used in the Normal likelihood
        '''
        T = self.settings["T"]
        Sigma = torch.zeros((K, T, T))
        for k in range(K):
            if self.cov_type == "diag" or T==1:
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].diag_embed(), \
                    sigma_chol).add(torch.eye(T))

            if self.cov_type == "full" and T > 1:
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].diag_embed(), \
                    sigma_chol[k]).add(torch.eye(T))

        return Sigma

