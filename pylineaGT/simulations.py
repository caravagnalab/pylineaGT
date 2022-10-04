import torch
import pyro.distributions as distr
import pyro


class Simulate():
    def __init__(self, seed, N, T, K, max_value=4000, 
        var_loc=110, var_scale=195, min_var=20, eta=1, cov_type="full",
        label="", max_iter=100, alpha=.35):

        self.settings = {"N":N, "T":T, "K":K, 
            # "mean_loc":torch.tensor(mean_loc).float(), 
            "max_value":torch.tensor(max_value).float(),
            "var_loc":torch.tensor(var_loc).float(), 
            "var_scale":torch.tensor(var_scale).float(), 
            "min_var":torch.tensor(min_var).float(),
            "eta":torch.tensor(eta).float(),
            "slope":torch.tensor(0.17).float(), \
            "intercept":torch.tensor(24.24).float(),
            # "slope":0.09804862, "intercept":22.09327233,
            "alpha":alpha, "seed":seed}

        self._max_iter = max_iter

        self.cov_type = cov_type
        self.label = label

    #     self.set_sigma_constraints()


    # def set_sigma_constraints(self):
    #     slope = self.settings["slope"]
    #     intercept = self.settings["intercept"]

    #     T = self.settings["T"]
    #     slope_tns = torch.repeat_interleave(slope, T)
    #     intercept_tns = torch.repeat_interleave(intercept, T)
    #     self.lm = {"slope":slope_tns, "intercept":intercept_tns}
    #     self.settings["slope"] = slope_tns
    #     self.settings["intercept"] = intercept_tns


    def generate_dataset(self):
        pyro.set_rng_seed(self.settings["seed"])

        K = self.settings["K"]
        T = self.settings["T"]
        N = self.settings["N"]
        max_value = self.settings["max_value"]
        # mean_loc = self.settings["mean_loc"]
        var_loc = self.settings["var_loc"]
        var_scale = self.settings["var_scale"]
        min_var = self.settings["min_var"]
        eta = self.settings["eta"]

        weights = distr.Dirichlet(torch.ones(K)).sample()
        
        z = torch.zeros((N,), dtype=torch.long)
        x = torch.zeros((N,T))

        while len(z.unique()) < K:
            for n in pyro.plate("assign", N):
                z[n] = distr.Categorical(weights).sample()

        mean = torch.zeros(K,T)
        sigma_vector = torch.zeros(K,T)
        var_constr = torch.zeros(K,T)
        sigma_chol = torch.zeros((K,T,T))
        
        for k in pyro.plate("clusters", K):

            for t in pyro.plate("timepoints", T):
                mean[k,t] = distr.Uniform(0, max_value).sample()
                # mean[k,t] = distr.Normal(mean_loc, mean_scale).sample()
                sigma_vector[k,t] = distr.Normal(var_loc, var_scale).sample()

                # check for negative values
                # while mean[k,t] < 0:
                #     mean[k,t] = distr.Normal(mean_loc, mean_scale).sample()

                var_constr[k,t] = mean[k,t] * self.settings["slope"] + self.settings["intercept"]

                # check for negative values
                while sigma_vector[k,t] < min_var or sigma_vector[k,t] > var_constr[k,t]:
                    sigma_vector[k,t] = distr.Normal(var_loc, var_scale).sample()
            
            if self.cov_type == "full" and T>1:
                sigma_chol[k] = distr.LKJCholesky(T, eta).sample()

        if self.cov_type == "diag" or T==1:
            sigma_chol = torch.eye(T) * 1.

        mean, sigma_vector, var_constr = self._check_means(mean, sigma_vector, var_constr)

        Sigma = self._compute_Sigma(sigma_chol, sigma_vector, K)
        for n in pyro.plate("obs", N):
            x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
            while torch.any(x[n,:] < 0):
                x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()

        self.dataset = x
        self.params = {"weights":weights, "mean":mean, "sigma":sigma_vector, \
            "var_constr":var_constr, "sigma_chol":sigma_chol, "z":z}

        self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), str(self.label)])


    def _check_means(self, mean, sigma_vector, var_constr, n=1):
        '''
        Function to perform a check in the sampled means, to avoid overlapping 
        distributions that by construction can't be distinguished.
        '''

        overlap = False

        if n == self._max_iter:
            print("MAX ITERATION")
            return mean, sigma_vector, var_constr

        max_value = self.settings["max_value"]
        
        min_var = self.settings["min_var"]
        var_loc = self.settings["var_loc"]
        var_scale = self.settings["var_scale"]

        slope = self.settings["slope"]
        intercept = self.settings["intercept"]
        # K = self.settings["K"]
        # T = self.settings["T"]

        for kk1 in range(self.settings["K"]):
            mu1 = mean[kk1,:]
            sigma1 = sigma_vector[kk1,:]

            for kk2 in range(kk1+1, self.settings["K"]):

                mu2 = mean[kk2,:]
                sigma2 = sigma_vector[kk2,:]
                constr2 = var_constr[kk2,:]

                resample = self._do_resample(mu1, sigma1, mu2, sigma2)

                while resample:
                    for tt in pyro.plate("time", self.settings["T"]):
                        mu2[tt] = distr.Uniform(0, max_value).sample()
                        sigma2[tt] = distr.Normal(var_loc, var_scale).sample()
                        constr2[tt] = mu2[tt] * slope + intercept
                        while sigma2[tt] < min_var or sigma2[tt] >= constr2[tt]:
                            sigma2[tt] = distr.Normal(var_loc, var_scale).sample()
                        
                        # sigma2[tt] = mu2[tt] * slope + intercept - \
                        #     distr.Normal(0,10).sample().abs()
                        # if sigma2[tt] < min_var: sigma2[tt] = min_var

                        resample = self._do_resample(mu1, sigma1, mu2, sigma2)
                        if not resample: break

                mean[kk2,:] = mu2
                sigma_vector[kk2,:] = sigma2
                var_constr[kk2,:] = constr2
                
                # check on the previously 
                for kk_tmp in range(kk2):
                    tmp_overlap = self._do_resample(mean[kk_tmp,:], sigma_vector[kk_tmp,:], mu2, sigma2)

                    if tmp_overlap:
                        overlap = True

        while overlap:
            return self._check_means(mean, sigma_vector, var_constr, n=n+1)

        return mean, sigma_vector, var_constr


    def _do_resample(self, mu, sigma, mu_k, sigma_k):
        # if they are separated even only in one timepoints, I do not need to resample
        alpha = self.settings["alpha"]

        for tt in range(self.settings["T"]):
            norm1 = distr.Normal(mu_k[tt], sigma_k[tt])
            norm2 = distr.Normal(mu[tt], sigma[tt])

            lower_q1 = norm1.icdf(torch.tensor(alpha))
            upper_q1 = norm1.icdf(torch.tensor(1-alpha))

            lower_q2 = norm2.icdf(torch.tensor(alpha))
            upper_q2 = norm2.icdf(torch.tensor(1-alpha))

            if (lower_q1 > upper_q2) or (upper_q1 < lower_q2):
                return False

        return True


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

