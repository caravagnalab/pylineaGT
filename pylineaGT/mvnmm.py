from collections import defaultdict
from pyclbr import Function
from tokenize import Double
from typing import Dict
from xmlrpc.client import Boolean
import pyro
import pyro.distributions as distr
import torch
import numpy as np
import random
import sklearn.metrics

from pyro.infer import SVI
from torch.distributions import constraints
from sklearn.cluster import KMeans
from tqdm import trange


class MVNMixtureModel():
    def __init__(self, K, data, IS=[], columns=[], lineages=[]):
        self._set_dataset(data)
        self.IS = np.array(IS)
        self.dimensions = columns # `columns` will be a list of type ["early.MNC", "mid.MNC", "early.Myeloid", "mid.Myeloid"]
        self.lineages = lineages

        self.K, self._N, self._T = K, self.dataset.shape[0], self.dataset.shape[1]
        self._initialize_attributes()
        self._initialize_sigma_constraint() 


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
            "mean_loc":self.dataset.float().mean(), "var_scale":300, "eta":1}
        self._autoguide = False
        self._enumer = "parallel"

        if len(self.dimensions) == 0:
            self.dimensions = [str(_) for _ in range(self._T)]


    def filter_dataset(self, min_cov=50, n=None, min_ccf=.05, k_interval=(5,25),
            metric="calinski_harabasz_score", random_state=25):
        '''
        Function to filter the input dataset.
        - `min_cov` -> hard threshold for each observation. Only the observations with at least a 
        coverage of `thr` across the timepoints are kept.
        - `n` -> number of random observations to draw from the dataset.
        - `min_ccf` -> percentage of the cumulative distribution that will be used as a threshold 
        for each timepoint to remove observations with a too low coverage. 
        It will perform a K-means, for `k` in `k_interval`, to search for the best `K` (optimizing 
        the input metric), and performing K-means to discard observations belonging to clusters 
        with the centroid of all timepoints below `5%` of the sum of centroids for all the timepoints.
        - `k_interval` -> interval of `K` values to look for the best `K`.
        - `metric` -> metric used to retrieve the best `K`, among `calinski_harabasz_score` and `silhouette`.
        - `random_state` -> seed value.
        '''
        try: self.IS = self.IS[self.dataset.sum(dim=1) >= min_cov]
        except: pass
        finally: self.dataset = self.dataset[self.dataset.sum(dim=1) >= min_cov,]
        
        self._initialize_sigma_constraint()

        self.dataset, self.IS = self._filter_dataframe_init(min_ccf=min_ccf, k_interval=k_interval, \
            metric=metric, random_state=random_state)

        if n is not None:  # takes a random sample from the dataset
            n = min(n, self.dataset.shape[0])
            np.random.seed(random_state)
            idx = np.random.randint(self.dataset.shape[0], size=n)
            try: self.IS = self.IS[idx]
            finally: self.dataset = self.dataset[idx,:]

        self._N = self.dataset.shape[0]
        self.params["N"] = self._N
        self.init_params["N"] = self._N


    def _initialize_sigma_constraint(self):
        '''
        Function to initialize the constraints on the variance for each dimension in each cluster. 
        It performs a linear regression on the marginal distribution of each dimension and performs 
        a check on the x-intercept, to avoid negative values of y for x in [0,max_cov].
        '''
        self.lm = dict()
        slope, intercept = torch.zeros((self._T)), torch.zeros((self._T))
        for t in range(self._T):
            xx, yy = np.unique(self.dataset[:,t], return_counts=True)
            lm = sklearn.linear_model.LinearRegression()
            fitted = lm.fit(xx.reshape(-1,1), yy.reshape(-1,1))  # eatimate the coefficient of the linear reg
            slope[t] = torch.tensor(float(fitted.coef_[0]))
            intercept[t] = torch.tensor(fitted.intercept_[0])
            
            # check that y=0 when x >= (max_cov*1.5 + max_cov/10)
            if (slope[t] * torch.tensor(max(xx) * 1.5) + intercept[t]) <= (max(xx)/10):
                intercept[t] = torch.max(-1*( slope[t] * torch.tensor(max(xx)*1.5) ), \
                    torch.tensor(max(xx) / 10))
        self.lm["slope"], self.lm["intercept"] = slope, intercept


    def _filter_dataframe_init(self, min_ccf=.05, k_interval=(5,25), \
            metric="calinski_harabasz_score", random_state=25):
        '''
        Function to filter the input dataset according to the centroid the clusters output
        from a KMeans, with `K` being the best `K` in `k_interval` according to `metric`.
        - `min_ccf` -> percentage of the cumulative distribution that will be used as a threshold 
        for each timepoint to remove observations with a too low coverage. 
        It will perform a K-means, for `k` in `k_interval`, to search for the best `K` (optimizing 
        the input metric), and performing K-means to discard observations belonging to clusters 
        with the centroid of all timepoints below `5%` of the sum of centroids for all the timepoints.
        - `k_interval` -> interval of `K` values to look for the best `K`.
        - `metric` -> metric used to retrieve the best `K`, among `calinski_harabasz_score` and `silhouette`.
        - `random_state` -> seed value.
        '''
        index_fn = self._find_index_function(metric)
        N, K = self.dataset.shape[0], self._find_best_k(k_interval=k_interval, index_fn=index_fn, random_state=random_state)
        km = KMeans(n_clusters=K).fit(self.dataset)
        assert km.n_iter_ < km.max_iter

        clusters = km.labels_
        ctrs = torch.tensor(km.cluster_centers_).float().detach() + torch.abs(torch.normal(0, 1, (K, self._T)))
        keep = torch.where((ctrs / ctrs.sum(dim=0) > min_ccf).sum(dim=1) > 0)[0]
        
        try: ii = self.IS[np.in1d(np.array(clusters), keep)]
        except: ii = self.IS
        finally: return self.dataset[np.in1d(np.array(clusters), keep)], ii


    def _find_index_function(self, index="calinski_harabasz_score"):
        if index == "calinski_harabasz_score":
            return sklearn.metrics.calinski_harabasz_score
        if index == "silhouette":
            return sklearn.metrics.silhouette_score


    def _find_best_k(self, k_interval=(2,30), index_fn=sklearn.metrics.calinski_harabasz_score, random_state=25):
        k_interval = (k_interval[0], max(k_interval[1], self.K))
        scores = torch.zeros(k_interval[1])

        for k in range(k_interval[0], k_interval[1]):
            km = KMeans(n_clusters=k, random_state=random_state)
            labels = km.fit_predict(self.dataset)
            scores[k] = index_fn(self.dataset, labels)
        best_k = scores.argmax()
        return best_k


    def model(self):
        N, K = self._N, self.K

        weights = pyro.sample("weights", distr.Dirichlet(torch.ones(K)))  # mixing proportions for each component sample the mixing proportion
        
        mean_scale = self.hyperparameters["mean_scale"]
        mean_loc = self.hyperparameters["mean_loc"]
        var_scale = self.hyperparameters["var_scale"]
        eta = self.hyperparameters["eta"]
        var_constr = self.init_params["var_constr"]

        with pyro.plate("time_plate", self._T):
            with pyro.plate("comp_plate", K):
                mean = pyro.sample("mean", distr.Normal(mean_loc, mean_scale))

        with pyro.plate("time_plate2", self._T):
            with pyro.plate("comp_plate3", K):
                variant_constr = pyro.sample("var_constr", distr.Delta(var_constr))
                sigma_vector = pyro.sample("sigma_vector", distr.HalfNormal(var_scale))

        if self.cov_type == "diag":
            sigma_chol = torch.eye(self._T) * 1.
        if self.cov_type == "full":
            with pyro.plate("comp_plate2", K):
                sigma_chol = pyro.sample("sigma_chol", distr.LKJCholesky(self._T, eta))

        Sigma = self.compute_Sigma(sigma_chol, sigma_vector, K)

        with pyro.plate("data_plate", N):
            z = pyro.sample("z", distr.Categorical(weights), infer={"enumerate":"parallel"})
            x = pyro.sample("obs", distr.MultivariateNormal(loc=mean[z], \
                scale_tril=Sigma[z]), obs=self.dataset)


    def guide(self) -> Function:
        '''
        Function to define the guide for the model.
        - If `self._autoguide` is `True`, the function will return the `AutoDelta` function.
        - Otherwise, it will return the explicit guide. In that case, also the `enumer` 
          flag will be taken into consideration.
        '''
        if not self._autoguide:
            def guide_expl():
                params = self._initialize_params(random_state=self._seed)
                N, K = params["N"], params["K"]

                weights_param = pyro.param("weights_param", lambda: params["weights"], \
                    constraint=constraints.simplex)
                mean_param = pyro.param("mean_param", lambda: params["mean"], \
                    constraint=constraints.positive)
                sigma_chol_param = pyro.param("sigma_chol_param", lambda: params["sigma_chol"], \
                    constraint=constraints.corr_cholesky)
                
                weights = pyro.sample("weights", distr.Delta(weights_param).to_event(1))
                with pyro.plate("time_plate", self._T):
                    with pyro.plate("comp_plate", K):
                        mean = pyro.sample("mean", distr.Delta(mean_param))

                with pyro.plate("time_plate2", self._T):
                    with pyro.plate("comp_plate3", K):
                        variant_constr = pyro.sample(f"var_constr", distr.Delta(params["var_constr"]))
                        sigma_vector_param = pyro.param(f"sigma_vector_param", lambda: params["sigma_vector"], 
                            # constraint=constraints.positive)
                            constraint=constraints.interval(0, variant_constr))
                        sigma_vector = pyro.sample(f"sigma_vector", distr.Delta(sigma_vector_param))
                
                if self.cov_type == "full":
                    with pyro.plate("comp_plate2", K):
                        sigma_chol = pyro.sample("sigma_chol", distr.Delta(sigma_chol_param).to_event(2))

                # z_param = pyro.param("z_param", lambda: torch.ones(N, K) / K, \
                #     constraint=constraints.simplex) 
                with pyro.plate("data_plate", N):
                    z = pyro.sample("z", distr.Categorical(weights), \
                        infer={"enumerate":self._enumer})
            return guide_expl
        else:
            raise NotImplementedError


    def compute_Sigma(self, sigma_chol, sigma_vector, K):
        Sigma = torch.zeros((K, self._T, self._T))
        for k in range(K):
            if self.cov_type == "diag":
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].sqrt().diag_embed(), \
                    sigma_chol).add(torch.eye(self._T))
            if self.cov_type == "full":
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].sqrt().diag_embed(), \
                    sigma_chol[k]).add(torch.eye(self._T))
        return Sigma


    def _initialize_params(self, random_state=25):
        '''
        Function to initialize the parameters.
        It performs a K-means clustering and define the initial weights 
        as the proportion of IS assigned to each cluster and the
        Poisson lambda or Gaussian mean as the centers of each cluster.
        '''
        N, K = self.params["N"], self.params["K"]
        if not self.init_params["is_computed"]:
            km = KMeans(n_clusters=K, random_state=random_state).fit(self.dataset)
            self.init_params["clusters"] = torch.from_numpy(km.labels_)
            self.init_params["z_assignments"] = torch.zeros(N, K)

            for n in range(N):
                self.init_params["z_assignments"][n, self.init_params["clusters"][n]] = 1
            
            w = torch.tensor([(np.where(km.labels_ == k)[0].shape[0]) / N for k in range(km.n_clusters)])
            self.init_params["weights"] = w.float().detach()

            ctrs = torch.tensor(km.cluster_centers_).float().detach() + torch.abs(torch.normal(0, 1, (K, self._T)))
            ctrs[ctrs <= 0] = 0.01

            var = torch.zeros(self.params["K"], self._T)
            self.init_params["var_constr"] = torch.zeros(self.params["K"], self._T)
            for cl in torch.unique(self.init_params["clusters"]):
                var[cl,:] = torch.var(self.dataset[torch.where(self.init_params["clusters"]==cl)].float(), \
                    dim=0, unbiased=False)
                # self.init_params["var_constr"][cl,:] = 100
                self.init_params["var_constr"][cl,:] = ctrs[cl,:] * self.lm["slope"] + self.lm["intercept"]

            var += torch.abs(torch.normal(0, 1, (K, self._T)))
            var[var > self.init_params["var_constr"]] = self.init_params["var_constr"][var > self.init_params["var_constr"]] - .1
            # var[var > 1000] = 1000. - .1
            var[var < 1] = 1.

            self.init_params["mean"] = ctrs
            self.init_params["sigma_vector"] = var

            if self.cov_type == "diag":
                self.init_params["sigma_chol"] = torch.eye(self._T) * 1.
            if self.cov_type == "full":
                self.init_params["sigma_chol"] = torch.zeros((K, self._T, self._T))
                for k in range(K):
                    self.init_params["sigma_chol"][k,:,:] = distr.LKJCholesky(self._T, \
                        self.hyperparameters["eta"]).sample()
            
            self.init_params["sigma"] = self.compute_Sigma(sigma_chol=self.init_params["sigma_chol"],\
                sigma_vector=self.init_params["sigma_vector"], K=self.init_params["K"])
            
            self.init_params["is_computed"] = True
        return self.init_params


    def fit(self, steps=500, optim_fn=pyro.optim.Adam, lr=0.001, cov_type="diag", \
            loss_fn=pyro.infer.TraceEnum_ELBO(), convergence=False, initializ=True, \
            min_steps=1, p=0.05, random_state=25, show_progr=True):
        pyro.enable_validation(True)
        pyro.clear_param_store()

        self._settings = {"optim":optim_fn({"lr":lr, "betas": (0.93, 0.999)}), "loss":loss_fn, "lr":lr}
        self._is_trained = False
        self._max_iter = steps
        self.cov_type = cov_type
        self._seed = random_state

        if initializ:
            # set the guide and initialize the SVI object to minimize the initial loss 
            loss, self._seed = min((self._initialize(seed), seed) for seed in range(100))
            self._initialize(self._seed)
        else:
            # set the guide and intialize the SVI object
            self._global_guide = self.guide()
            self.svi = SVI(self.model, self._global_guide, \
                        optim=self._settings["optim"], loss=self._settings["loss"])
            self.svi.loss(self.model, self._global_guide)

        losses_grad = self._train(steps=self._max_iter, convergence=convergence, \
            min_steps=min_steps, p=p, show_progr=show_progr)

        self._is_trained = True
        self.losses_grad_train = losses_grad  # store the losses and the gradients for weights/lambd
        self.guide_trained = self._global_guide  # store the trained guide and model
        self.model_trained = self.model
        self.params = self._get_learned_parameters()
        return


    def _initialize(self, seed) -> Double:
        '''
        Function used to optimize the initialization of the SVI object.
        '''
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        self._global_guide = self.guide()
        self.svi = SVI(self.model, self._global_guide, optim=self._settings["optim"], \
            loss=self._settings["loss"]) 
        return self.svi.loss(self.model, self._global_guide)


    def _train(self, steps, convergence, min_steps=1, p=0.01, show_progr=True):
        '''
        Function to perform the training of the model. \\
        `steps` is the maximum number of steps to be performed. \\
        It checks for each step `t>=100` if the estimated `mean` and `sigma`
        parameters remain constant for at least `30` consecutive iterations. \\
        It returns a dictionary with the computed losses, parameters gradients
        and negative log-likelihood per iteration.
        '''
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            if name in ["weights_param", "mean_param", "sigma_vector_param"]:
                value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

        losses = list()
        mean_conv, sigma_conv = [self.init_params["mean"],self.init_params["mean"]], \
            [self.init_params["sigma_vector"],self.init_params["sigma_vector"]]  #to store the lambda at the previous and current iteration
        conv = 0

        t = trange(steps, desc='Bar desc', leave=True) if show_progr else range(steps)
        for step in t:
            self.iter = step
            elb = self.svi.step() / self._N
            losses.append(elb)

            params_step = self._get_learned_parameters() 

            # gradient_norms = self._reset_params(params=params_step, p=.01, gradient_norms=gradient_norms)
            if convergence and step >= min_steps:
                mean_conv[0], mean_conv[1] = mean_conv[1], params_step["mean"]
                sigma_conv[0], sigma_conv[1] = sigma_conv[1], params_step["sigma_vector"]
                conv = self._convergence(mean_conv, sigma_conv, conv, p=p)
                # conv = self._convergence_grads(gradient_norms, conv)
                if conv == 10:
                    break
            
            if show_progr:
                t.set_description("ELBO %f" % elb)
                t.refresh()
        return {"losses":losses, "gradients":dict(gradient_norms)}


    # def _reset_params(self, params=None, p=.01, gradient_norms=None):
    #     if params is None:
    #         params = self._get_learned_parameters()
    #     if random.random() < p:
    #         clusters = self._retrieve_cluster(params=params)  # current clusters
    #         means = torch.zeros_like(self.init_params["mean"])
    #         for cl in clusters.unique():
    #             d_k = self.dataset.index_select(dim=0, index=torch.where(clusters==cl)[0]).float()
    #             means[int(cl),] = d_k.mean(dim=0).clone().detach().requires_grad_()

    #         pyro.get_param_store()["mean_param"] = means.clone().detach().requires_grad_()
    #         for name, value in pyro.get_param_store().named_parameters():
    #             if name == "mean_param":
    #                 value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))
    #     return gradient_norms


    def _convergence_grads(self, gradient_norms, conv, p=0.005):
        for gr in gradient_norms.keys():
            if gr in ["mean_param", "sigma_vector_param"]:
                prev = gradient_norms[gr][-2]
                curr = gradient_norms[gr][-1]
                cc = np.abs(prev - curr) < curr*p
                # print(gradient_norms[gr][-2], gradient_norms[gr][-1], np.abs(prev - curr), curr*p)
        if cc:
            return conv +1
        return 0


    def _convergence(self, mean_conv, sigma_conv, conv, p):
        if self._check_convergence(mean_conv, p) and self._check_convergence(sigma_conv, p):
            return conv + 1
        return 0


    def _check_convergence(self, par, perc=0.01) -> Boolean:
        '''
        - `par` -> list of 2 elements given the previous (at index 0) and current (at index 1) 
        estimated values for a parameter.
        - `perc` -> numeric value in `[0,1]`.
        The function returns `True` if more than 95% of the values changed less than `perc*100`% 
        in the current step with respect to the previous one.
        '''
        n = 0
        for k in range(par[0].shape[0]):
            for t in range(par[0].shape[1]):
                p = perc * torch.max(torch.tensor(1), par[0][k,t])
                if torch.absolute(par[0][k,t] - par[1][k,t]) <= p:
                    n += 1
        return n >= .99 * self.params["K"]*self.params["T"]


    def _get_learned_parameters(self) -> Dict:
        '''
        Function to store and return the MAP estimate of the 
        parameters from the `pyro.param_store()`. \\
        It returns a dictionary 
        '''
        param_store = pyro.get_param_store()
        if self._autoguide:
            raise NotImplementedError
            # param_store = self._global_guide()
        
        p = {}
        p["N"], p["K"], p["T"] = self.params["N"], self.params["K"], self.params["T"]
        p["weights"] = param_store["weights_param"]

        p["mean"] = param_store["mean_param"]
        p["sigma_vector"] = param_store["sigma_vector_param"]
        p["sigma_chol"] = param_store["sigma_chol_param"]

        if self._is_trained:
            p["sigma"] = self.compute_Sigma(p["sigma_chol"], p["sigma_vector"], p["K"])
        return p


    def classifier(self, params=None, perc=.5, t=10) -> dict():
        '''
        Function to perform the classification of the observations based 
        on the computed assignment probabilities.
        '''
        if params is None:
            params = self.params
            flag = True
        
        params["z_probs"], params["z_assignments"] = self.compute_assignments(params=params)
        params["clusters"] = self._retrieve_cluster(params["z_assignments"])
        params = self._reduce_assignments(p=perc, params=params, t=t) 
        try:
            if flag:
                self.params = params
                self.K = params["K"]
        finally:
            return

    
    def _retrieve_cluster(self, assignments=None, params=None):
        if params is not None:
            _, assignments = self.compute_assignments(params=params)
        return assignments.argmax(dim=1)


    def _reduce_assignments(self, t=10, p=.5, params=None, min_prob=.20):
        '''
        Function to adjust the final assignments.
        '''
        if params.get("z_assignments", False) is False:
            params["z_probs"], params["z_assignments"] = self.compute_assignments(params)

        n = min(p/100 * params["N"], t)
        keep_low = torch.tensor(np.where( (params["z_assignments"].sum(dim=0) <= n) & \
            (params["z_assignments"].sum(dim=0) > 0) )[0])

        for cluster in keep_low:
            count = 0
            obs = torch.tensor(np.where(params["z_assignments"][:,cluster]==1)[0])
            for o in obs:
                if params["z_probs"][o, cluster] >= min_prob:
                    count += 1
            if count < obs.shape[0]:
                keep_low = torch.cat([keep_low[0:cluster], keep_low[cluster+1:]])

        keep = torch.tensor(np.where(params["z_assignments"].sum(dim=0) > n)[0])
        try:
            keep = torch.unique(torch.cat([keep, keep_low]))
        except:
            keep = keep
        
        if len(keep) == 0:
            return params
        
        params["K"] = len(keep)
        params.pop("z_assignments")
        params.pop("z_probs")
        for par in params.keys():
            if (self.cov_type=="full" and par in ["N", "K", "T", None]) or \
                (self.cov_type=="diag" and par in ["N", "K", "T", "sigma_chol", None]):
                continue
            params[par] = params[par].index_select(dim=0, index=keep)

        params["weights"] = params["weights"] / params["weights"].sum()
        params["z_probs"], params["z_assignments"] = self.compute_assignments(params=params)
        params["clusters"] = params["z_assignments"].max(1).indices
        return params


    def compute_assignments(self, params=None) -> torch.Tensor:
        '''
        Returns a matrix K x N with 0 or 1 for each x_kn, assigning for 
        each obs a 1 to the cluster with the highest posterior probability.
        '''
        if params is None:
            params = self.params
        N, K = params["N"], params["K"]

        if params.get("z_probs", False) is False:
            params["z_probs"] = self._compute_assignment_probs(params=params)
        
        params["z_assignments"] = torch.zeros(N, K)
        for n in range(N):
            assignment_idx = params["z_probs"][n].argmax(0)
            params["z_assignments"][n, assignment_idx] = 1
        return params["z_probs"], params["z_assignments"]


    def _log_lik(self, params=None) -> torch.Tensor:
        ''' 
        Returns a N-dim vector, with the log-likelihood of each obs, performing 
        the log of the sum (over K) of the exp of each weighted log-prob. 
        `log( P(x|mu,Sigma) ) = log( sum^K( exp( log(pi_k)+log(P(x_n|mu_k,Sigma_k)) ) ) ) `
        '''
        if params is None:
            params = self.params
        
        return self._logsumexp(self._weighted_log_prob(params))


    def _weighted_log_prob(self, params=None) -> torch.Tensor:
        '''
        Returns a matrix K x N, by adding the log-weights of each cluster to the 
        log-probs and assuming indipendence among the d-dimensions of the dataset
        (summing the log-probs).
        `log( pi_k ) + log( P(x_n|mu_k,Sigma_k) ) `
        '''
        # We sum the log-likelihoods for each timepoint -> K x N matrix
        # And we sum the weight for the corresponding distribution
        if params is None:
            params = self.params

        return self.compute_log_prob(params) + torch.log(params["weights"]).reshape((params["K"],1))


    def compute_log_prob(self, params=None) -> torch.Tensor:
        '''
        Returns a matrix of dimensions K x N x T. 
        For each cluster, there is a N x T matrix with the log-probabilities 
        for each observation `x_n`, given the parameters `mu_k` and `Sigma_k`.
        '''
        if params is None:
            params = self.params
        
        N, K = params["N"], params["K"]
        
        Sigma = self.compute_Sigma(params["sigma_chol"], params["sigma_vector"], params["K"])
        logprob = torch.zeros(K, N)
        for k in range(K):  # compute it per each cluster, observation and timepoint
            logprob[k,:] = distr.MultivariateNormal(loc=params["mean"][k], \
                scale_tril=Sigma[k]).log_prob(self.dataset)
        
        self.logprob = logprob
        return self.logprob


    def _compute_assignment_probs(self, params=None) -> torch.Tensor:
        '''
        Returns a matrix N x K with the posterior probabilities for each obs n to be assigned
        to cluster k.
        `P(z_nk=c|x) = (pi_c*p(x|mu_c,Sigma_c)) / sum^K( pi_k*p(x|mu_k,Sigma_k) ) \
                     = exp(loglik_c) / exp(loglik) =  exp( loglik_c - loglik )`

        If specific parameters are give as input, it uses those values.
        Otherwise, it uses the estimated paramters.
        '''
        if params is None:
            params = self.params
        
        params["z_probs"] = torch.exp(self._weighted_log_prob(params) - \
            self._log_lik(params)).transpose(dim0=1, dim1=0)
        return params["z_probs"]


    def _logsumexp(self, weighted_lp) -> torch.Tensor:
        '''
        Returns `m + log( sum( exp( weighted_lp - m ) ) )`
        - `m` is the the maximum value of weighted_lp for each observation among the K values
        - `torch.exp(weighted_lp - m)` to perform some sort of normalization
        In this way the `exp` for the maximum value will be exp(0)=1, while for the 
        others will be lower than 1, thus the sum across the K components will sum up to 1.
        '''
        m = torch.amax(weighted_lp, dim=0)  # the maximum value for each observation among the K values
        summed_lk = m + torch.log(torch.sum(torch.exp(weighted_lp - m), axis=0))
        return summed_lk


    def _n_parameters(self, params=None) -> int:
        ''' Return the number of free parameters in the model. '''
        if params is None:
            params = self.params

        if self.cov_type == "full":
            return params["weights"].numel() + params["mean"].numel() \
                + params["sigma_vector"].numel() + self._T*(self._T-1) / 2
        if self.cov_type == "diag":
            return params["weights"].numel() + params["mean"].numel() \
                + params["sigma_vector"].numel()


    def compute_ic(self, method=None, params=None) -> np.array:
        if params is None:
            params = self.params

        self.nll = self._compute_nll(params=params)
        if method == "BIC":
            return self._compute_bic(params=params, nll=self.nll)
        if method == "AIC":
            return self._compute_aic(params=params, nll=self.nll)
        if method == "ICL":
            return self._compute_icl(params=params, nll=self.nll)
        if method == "NLL":
            return self.nll
        if method is None:
            return np.array([self._compute_bic(params=params, nll=self.nll),\
                self._compute_aic(params=params, nll=self.nll), \
                self._compute_icl(params=params, nll=self.nll), self.nll])


    def _compute_nll(self, params=None):
        nll = -(self._log_lik(params=params).sum())
        return nll.detach()


    def _compute_bic(self, params=None, nll=None): # returns a np.float64 
        if nll is None:
            nll = self._compute_nll(params=params)
        if params is None:
            params = self.params
        return 2*nll + self._n_parameters(params=params) * np.log(params["N"])


    def _compute_aic(self, params=None, nll=None):
        if nll is None:
            nll = self._compute_nll(params=params)
        return 2*nll + 2*self._n_parameters(params=params)


    def _compute_icl(self, params=None, nll=None):
        if nll is None:
            nll = self._compute_nll(params=params)
        return self._compute_bic(params=params) + self._compute_entropy(params=params)


    def _compute_entropy(self, params=None) -> np.array:
        '''
        `entropy(z) = - sum^K( sum^N( z_probs_nk * log(z_probs_nk) ) )`
        `entropy(z) = - sum^K( sum^N( exp(log(z_probs_nk)) * log(z_probs_nk) ) )`
        '''
        if params is None:
            params = self.params

        logprobs = self._weighted_log_prob(params=params) - self._log_lik(params=params)  # log(z_probs)
        entr = 0
        for n in range(params["N"]):
            for k in range(params["K"]):
                entr += torch.exp(logprobs[k,n]) * logprobs[k,n]
        return -entr.detach()
