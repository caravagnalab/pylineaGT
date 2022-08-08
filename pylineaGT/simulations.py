from .run import run_inference
import torch
import pyro.distributions as distr
import pickle
import os
import pyro

class Simulate():
    def __init__(self, N=200, T=5, K=15, 
        mean_loc=400, mean_scale=1500, 
        var_loc=50, var_scale=30, min_var=5,
        eta=1, cov_type="full", label="", seed=None):

        self.settings = {"N":N, "T":T, "K":K, 
            "mean_loc":torch.tensor(mean_loc).float(), 
            "mean_scale":torch.tensor(mean_scale).float(),
            "var_loc":torch.tensor(var_loc).float(), 
            "var_scale":torch.tensor(var_scale).float(), 
            "min_var":torch.tensor(min_var).float(),
            "eta":torch.tensor(eta).float(),
            "seed":seed}

        pyro.set_rng_seed(seed)

        self.cov_type = cov_type
        self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), str(label)])

        self.set_sigma_constraints()


    def set_sigma_constraints(self, slope=0.09804862, intercept=22.09327233):
        T = self.settings["T"]
        slope_tns = torch.repeat_interleave(torch.tensor(slope), T)
        intercept_tns = torch.repeat_interleave(torch.tensor(intercept), T)
        self.lm = {"slope":slope_tns, "intercept":intercept_tns}


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
        
        mean = torch.zeros(K,T)
        sigma_vector = torch.zeros(K,T)
        var_constr = torch.zeros(K,T)
        for k in range(K):
            mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))
            sigma_vector[k,:] = distr.Normal(var_loc, var_scale).sample(sample_shape=(T,))

            while torch.any(mean[k,:] < 0):
                mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))

            while torch.any(sigma_vector[k,:] < min_var):
                sigma_vector[k,:] = distr.Normal(var_loc, var_scale).sample(sample_shape=(T,))

        for k in range(K):
            var_constr[k,:] = mean[k,:] * self.lm["slope"] + self.lm["intercept"]

        sigma_vector[sigma_vector > var_constr] = var_constr[sigma_vector > var_constr] - .1

        if self.cov_type == "diag" or T==1:
            sigma_chol = torch.eye(T) * 1.
        
        if self.cov_type == "full" and T>1:
            sigma_chol = distr.LKJCholesky(T, eta).sample(sample_shape=(K,))

        Sigma = self._compute_Sigma(sigma_chol, sigma_vector, K)

        z = torch.zeros((N,), dtype=torch.long)
        x = torch.zeros((N,T))
        for n in range(N):
            z[n] = distr.Categorical(weights).sample()
            x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
            while torch.any(x[n,:] < 0):
                x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()

        self.dataset = x
        self.params = {"weights":weights, "mean":mean, "sigma_vector":sigma_vector, "z":z}


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


    # def _add_noise(self, x):
    #     noise_loc = self.settings["noise_loc"]
    #     noise_scale = self.settings["noise_scale"]
    #     N = self.settings["N"]
    #     T = self.settings["T"]

    #     noise = torch.normal(mean=torch.full((N,T),noise_loc), std=torch.full((N,T),noise_scale)).abs()

    #     return torch.ceil(x.add_(noise)).int()

    # def run_inference_sim(self, sim_id, n_runs):
    #     k_interval = [max(self.settings["K"]-5, 1), self.settings["K"]+5]
        
    #     for run in range(n_runs):
    #         new_sim_id = sim_id + "." + str(run)
    #         new_sim = run_inference(self.dataset, lineages=[], k_interval=k_interval, n_runs=n_runs)
    #     return inference



def generate_synthetic_data(N_values, T_values, K_values, n_datasets=1, check_present=True):
    files_list = [f for f in os.listdir('.') if os.path.isfile(f)]

    seeds = torch.randint(low=0, high=100, size=(n_datasets,))

    for n_df in range(n_datasets):
        for N in N_values:
            for T in T_values:
                for K in K_values:
                    mean_loc = 50
                    mean_scale = 500
                    var_scale = 400
                    sim = Simulate(N, T, K, mean_loc, mean_scale, var_scale, label=n_df, seed=seeds[n_df])

                    # check if the file is already present
                    if sim.sim_id+".data.pkl" in files_list and check_present:
                        continue

                    sim.generate_dataset() 
                    # save the file in the current directory
                    with open(sim.sim_id+".data.pkl", 'wb') as sim_file:
                        pickle.dump(sim, sim_file)

