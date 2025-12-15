import torch
import pyro.distributions as distr
import pyro


class Simulate():
    def __init__(self, seed, N, T, K, likelihood="MVN", mean_loc=500,
        var_loc=110, var_scale=195, min_var=20, eta=1, cov_type="full",
        g_alpha=2., g_beta=0.05, label="", max_iter=100, alpha=0.1, c=3.):

        self.settings = {"N":N, "T":T, "K":K, 
            "mean_loc":torch.tensor(mean_loc).float(), 
            "var_loc":torch.tensor(var_loc).float(), 
            "var_scale":torch.tensor(var_scale).float(), 
            "min_var":torch.tensor(min_var).float(),
            "eta":torch.tensor(eta).float(),
            "g_alpha":torch.tensor(g_alpha).float(),
            "g_beta":torch.tensor(g_beta).float(),
            "alpha":alpha, "c":c, "seed":seed}

        self._max_iter = max_iter
        self.likelihood = likelihood
        self.cov_type = cov_type
        self.label = label


    def generate_dataset(self):
        pyro.set_rng_seed(self.settings["seed"])

        K = self.settings["K"]
        T = self.settings["T"]
        N = self.settings["N"]

        mean_loc = self.settings["mean_loc"]
        var_loc = self.settings["var_loc"]
        var_scale = self.settings["var_scale"]
        min_var = self.settings["min_var"]
        eta = self.settings["eta"]
        g_alpha = self.settings["g_alpha"]
        g_beta = self.settings["g_beta"]

        weights = distr.Dirichlet(torch.ones(K)).sample()

        z = torch.zeros((N,), dtype=torch.long)
        x = torch.zeros((N,T))

        while len(z.unique()) < K:
            for n in pyro.plate("assign", N):
                z[n] = distr.Categorical(weights).sample()

        mean = torch.zeros(K,T)
        sigma = torch.zeros(K,T)  # sigma if lik==MVN or alpha if lik==NB
        var_constr = torch.zeros(K,T)
        sigma_chol = torch.zeros((K,T,T))

        for k in pyro.plate("clusters", K):

            for t in pyro.plate("timepoints", T):
                # mean[k,t] = distr.Uniform(1, mean_loc).sample()
                # # check for negative values in the means
                # while mean[k,t] < 0:
                #     mean[k,t] = distr.Uniform(1, mean_loc).sample()

                mean[k, t] = self._sample_mean(mean_loc)
                sigma[k, t] = self._sample_sigma(var_loc, var_scale, min_var, g_alpha, g_beta)
                
                # if self.likelihood == "MVN":
                #     sigma_alpha[k,t] = distr.Normal(var_loc, var_scale).sample()

                #     # check for negative values
                #     while sigma_alpha[k,t] < min_var:
                #         sigma_alpha[k,t] = distr.Normal(var_loc, var_scale).sample()

                # elif self.likelihood == "NB":
                #     nb_alpha[k,t] = distr.LogNormal(2.0, 0.5).sample()

            if self.likelihood=="MVN" and self.cov_type == "full" and T>1:
                sigma_chol[k] = distr.LKJCholesky(T, eta).sample()

        if self.likelihood=="MVN" and self.cov_type == "diag" or T==1:
            sigma_chol = torch.eye(T) * 1.

        mean, sigma = self._check_means(mean, sigma)

        if self.likelihood == "MVN":
            # x[n,:] = self._sample_mvn_lik(mean, sigma, sigma_chol, z, K, N)
            Sigma = self._compute_Sigma(sigma_chol, sigma, K)
            for n in pyro.plate("obs", N):
                x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
                while torch.any(x[n,:] < 0):
                    x[n,:] = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()

        elif self.likelihood == "NB":
            # x[n,:] = self._sample_nb_lik(mean.clone().detach(), sigma.clone().detach(), z, N)
            alpha = mean**2 / sigma
            probs = mean / (alpha + mean.clone().detach())
            for n in pyro.plate("obs", N):
                x[n,:] = distr.NegativeBinomial(total_count=alpha[z[n]], probs=probs[z[n]]).sample()

        self.dataset = x
        self.params = {"weights":weights, "mean":mean, "sigma":sigma, \
            "var_constr":var_constr, "sigma_chol":sigma_chol, "z":z}

        self.sim_id = ".".join(["N"+str(N), "T"+str(T), "K"+str(K), str(self.label)])


    def _sample_mean(self, mean_loc):
        mean = distr.Uniform(1, mean_loc).sample()
        # check for negative values in the means
        while mean < 0:
            mean = distr.Uniform(1, mean_loc).sample()
        
        return mean

    def _sample_sigma(self, var_loc, var_scale, min_var, g_alpha, g_beta):
        if self.likelihood == "MVN":
            sigma = distr.Normal(var_loc, var_scale).sample()

            # check for negative values
            while sigma < min_var:
                sigma = distr.Normal(var_loc, var_scale).sample()

        elif self.likelihood == "NB":
            sigma = distr.Gamma(g_alpha, g_beta).sample()

        return sigma

    # def _sample_mvn_lik(self, mean, sigma_alpha, sigma_chol, z, K, N):
    #     Sigma = self._compute_Sigma(sigma_chol, sigma_alpha, K)
    #     for n in pyro.plate("obs", N):
    #         x = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
    #         while torch.any(x < 0):
    #             x = distr.MultivariateNormal(loc=mean[z[n]], scale_tril=Sigma[z[n]]).sample()
    #     return x

    # def _sample_nb_lik(self, mean, sigma, z, N):
    #     alpha = mean**2 / sigma
    #     probs = alpha / (alpha + mean.clone().detach())
    #     for n in pyro.plate("obs", N):
    #         x = distr.NegativeBinomial(total_count=alpha[z[n]], probs=probs[z[n]]).sample()
    #     return x


    def _check_means(self, mean, sigma_alpha, n=1):
        '''
        Function to perform a check in the sampled means, to avoid overlapping 
        distributions that by construction can't be distinguished.
        '''

        overlap = False

        if n == self._max_iter:
            print("MAX ITERATION")
            return mean, sigma_alpha

        # max_value = self.settings["max_value"]
        mean_loc = self.settings["mean_loc"]

        min_var = self.settings["min_var"]
        var_loc = self.settings["var_loc"]
        var_scale = self.settings["var_scale"]
        g_alpha = self.settings["g_alpha"]
        g_beta = self.settings["g_beta"]

        for kk1 in range(self.settings["K"]):
            mu1 = mean[kk1,:]
            sigma1 = sigma_alpha[kk1,:]

            for kk2 in range(kk1+1, self.settings["K"]):
                mu2 = mean[kk2,:]
                sigma2 = sigma_alpha[kk2,:]
                resample = self._do_resample(mu1, sigma1, mu2, sigma2)

                while resample:
                    for tt in pyro.plate("time", self.settings["T"]):
                        mu2[tt] = self._sample_mean(mean_loc)
                        # mu2[tt] = distr.Uniform(1, mean_loc).sample()
                        sigma2[tt] = self._sample_sigma(var_loc, var_scale, min_var, g_alpha, g_beta)
                        # sigma2[tt] = distr.Normal(var_loc, var_scale).sample()
                        # while sigma2[tt] < min_var:
                        #     sigma2[tt] = distr.Normal(var_loc, var_scale).sample()

                        resample = self._do_resample(mu1, sigma1, mu2, sigma2)
                        if not resample: break

                mean[kk2,:] = mu2
                sigma_alpha[kk2,:] = sigma2

                # check on the previously 
                for kk_tmp in range(kk2):
                    tmp_overlap = self._do_resample(mean[kk_tmp,:], sigma_alpha[kk_tmp,:], mu2, sigma2)

                    if tmp_overlap:
                        overlap = True

        while overlap:
            return self._check_means(mean, sigma_alpha, n=n+1)

        return mean, sigma_alpha


    def _do_resample(self, mu, sigma, mu_k, sigma_k):
        # if they are separated even only in one timepoints, I do not need to resample

        for tt in range(self.settings["T"]):
            if self.likelihood == "MVN":
                alpha = self.settings["alpha"]

                norm1 = distr.Normal(mu_k[tt], sigma_k[tt])
                norm2 = distr.Normal(mu[tt], sigma[tt])

                lower_q1 = norm1.icdf(torch.tensor(alpha))
                upper_q1 = norm1.icdf(torch.tensor(1-alpha))

                lower_q2 = norm2.icdf(torch.tensor(alpha))
                upper_q2 = norm2.icdf(torch.tensor(1-alpha))

                if (lower_q1 > upper_q2) or (upper_q1 < lower_q2):
                    return False

            elif self.likelihood == "NB":
                c = self.settings["c"]

                sd1 = torch.sqrt(mu_k + sigma_k)
                sd2 = torch.sqrt(mu + sigma)
                
                if torch.abs(mu_k[tt] - mu[tt]) >= c * (sd1[tt] + sd2[tt]):
                    return False   # need resample

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

