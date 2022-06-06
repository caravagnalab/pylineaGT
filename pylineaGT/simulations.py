from .run_inference import run_inference
import torch
import pyro.distributions as distr
import pickle

class Simulate():
    def __init__(self, N=200, T=5, K=15, mean_loc=50, mean_scale=1000, var_scale=100, var_constr=100, 
        noise_loc=0, noise_scale=1, eta=1, cov_type="diag", random_state=25):
        self.settings = {"N":N, "T":T, "K":K, "mean_loc":mean_loc, "mean_scale":mean_scale,
            "var_scale":var_scale, "eta":eta, "var_constr":var_constr, 
            "noise_loc":float(noise_loc), "noise_scale":float(noise_scale)}
        self.cov_type = cov_type
        self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), 
            "m_loc"+str(mean_loc), "m_scale"+str(mean_scale), "v_scale"+str(var_scale)])

    def generate_dataset(self):
        K = self.settings["K"]
        T = self.settings["T"]
        N = self.settings["N"]
        mean_scale = self.settings["mean_scale"]
        mean_loc = self.settings["mean_loc"]
        var_scale = self.settings["var_scale"]
        # var_constr = self.settings["var_constr"]
        eta = self.settings["eta"]

        weights = distr.Dirichlet(torch.ones(K)).sample()
        
        mean = torch.zeros(K,T)
        sigma_vector = torch.zeros(K,T)
        for k in range(K):
            mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))
            sigma_vector[k,:] = distr.HalfNormal(var_scale).sample(sample_shape=(T,))
            while torch.any(mean[k,:] < 0):
                mean[k,:] = distr.Normal(mean_loc, mean_scale).sample(sample_shape=(T,))
            while torch.any(sigma_vector[k,:] < 0):
                sigma_vector[k,:] = distr.HalfNormal(var_scale).sample(sample_shape=(T,))

        if self.cov_type == "diag":
            sigma_chol = torch.eye(T) * 1.
        if self.cov_type == "full":
            sigma_chol = distr.LKJCholesky(T, eta).sample(sample_shape=(K,))

        Sigma = self._compute_Sigma(sigma_chol, sigma_vector, K)

        z = torch.zeros((N,), dtype=torch.long)
        x = torch.zeros((N,T))
        for n in range(N):
            z[n] = distr.Categorical(weights).sample()
            x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
            while torch.any(x[n,:] < 0):
                x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()

        self.dataset = self._add_noise(x)
        self.params = {"weights":weights, "mean":mean, "sigma_vector":sigma_vector, "z":z}


    def _compute_Sigma(self, sigma_chol, sigma_vector, K):
        Sigma = torch.zeros((K, self.settings["T"], self.settings["T"]))
        for k in range(K):
            if self.cov_type == "diag":
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].sqrt().diag_embed(), \
                    sigma_chol).add(torch.eye(self.settings["T"]))
            if self.cov_type == "full":
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].sqrt().diag_embed(), \
                    sigma_chol[k]).add(torch.eye(self.settings["T"]))
        return Sigma


    def _add_noise(self, x):
        noise_loc = self.settings["noise_loc"]
        noise_scale = self.settings["noise_scale"]
        N = self.settings["N"]
        T = self.settings["T"]

        noise = torch.normal(mean=torch.full((N,T),noise_loc), std=torch.full((N,T),noise_scale)).abs()

        return torch.ceil(x.add_(noise)).int()

    def run_inference_sim(self):
        k_interval = [max(self.settings["K"]-5, 1), self.settings["K"]+5]
        inference = run_inference(self.dataset, lineages=[], k_interval=k_interval, n_runs=2)
        return inference


def generate_synthetic_data(N_range, T_range, K_range, mean_loc_range, mean_scale_range, var_scale_range):

    for N in N_range:
        for T in T_range:
            for K in K_range:
                for mean_loc in mean_loc_range:
                    for mean_scale in mean_scale_range:
                        for var_scale in var_scale_range:
                            sim = Simulate(N, T, K, mean_loc, mean_scale, var_scale) 
                            sim.generate_dataset() 
                            inf = sim.run_inference_sim()
                            
                            with open(sim.sim_id+".df.pkl", 'wb') as sim_file:
                                pickle.dump(sim, sim_file)

                            with open(sim.sim_id+".run.pkl", 'wb') as inf_file:
                                pickle.dump(inf, inf_file)
