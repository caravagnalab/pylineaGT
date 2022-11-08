from cmath import nan
from collections import defaultdict
from pyclbr import Function
from typing import Dict, Tuple
from xmlrpc.client import Boolean

import pyro
import pyro.distributions as distr
import torch
import numpy as np
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

        self._input_K = K

        try: 
            assert self.dataset.unique(dim=0).shape[0] >= K
            self.error = False
        except: 
            print("The number of unique observations is smaller than the input number of clusters!")
            self.error = True


    def _set_dataset(self, data):
        if isinstance(data, torch.Tensor):
            self.dataset = data.int()
        else:  # convert the dataset to torch.tensor
            try:
                try: self.dataset = torch.tensor(data.values)  # if it's a Pandas dataframe
                except: pass
                try: self.dataset = torch.tensor(data)  # if it's a numpy array
                except: pass
            except: print("The input dataset must be a torch.Tensor, numpy.ndarray or pandas.DataFrame object!")

            # to ensure the correct dimensionality
            try: assert len(self.dataset.shape) > 1
            except: self.dataset.unsqueeze_(1)


    def _initialize_attributes(self):
        self.params = {"N":self._N, "K":self.K, "T":self._T}
        
        self.init_params = {"N":self._N, "K":self.K, "T":self._T, \
            "sigma":None, "mean":None, "weights":None, \
            "clusters":None, "var_constr":None, "is_computed":None}
        
        self.hyperparameters = { \
            "mean_scale":self.dataset.float().var(dim=0).max(), \
            # "mean_scale":min(self.dataset.float().var(), torch.tensor(1000).float()), \
            "mean_loc":self.dataset.float().max() / 2, \
            
            # mean and sd for the Normal prior of the variance
            "var_loc":torch.tensor(110).float(), \
            "var_scale":torch.tensor(195).float(), \
            "min_var":torch.tensor(5).float(), \
            
            "eta":torch.tensor(1).float(), \
            
            # slope and intercepts for the variance constraints
            # "slope":torch.tensor(0.299).float(), \
            # "intercept":torch.tensor(11.182).float()}
            "slope":torch.tensor(0.17).float(), \
            "intercept":torch.tensor(24.34).float()}
            # "slope":0.09804862, "intercept":22.09327233}

        self._autoguide = False
        self._enumer = "parallel"

        if len(self.dimensions) == 0:
            self.dimensions = [str(_) for _ in range(self._T)]


    def print_hyperparameters(self):
        for k, v in self.hyperparameters.items():
            print(f"{k} = {v}")


    def set_hyperparameters(self, name, value):
        '''
        Function to set the values of the hyperparameters. \\
        `name` -> a string among 
            - `mean_loc`, the center of the mean prior,
            - `mean_loc`, the variance of the mean prior, 
            - `var_loc`, the  center of the variance prior,
            - `var_scale`, the variance of the variance prior,
            - `max_var`, the maximum value attributed to the variance,
            - `min_var` the minimum value attributed to the variance.
        `value` -> either an integer or a floatin point number.
        '''
        self.hyperparameters[name] = torch.tensor(value).float()


    def filter_dataset(self, min_cov=0, min_frac=.05, k_interval=(5,15), n=None,
            metric="calinski_harabasz_score", seed=5):
        '''
        Function to filter the input dataset.
        - `min_cov` -> hard threshold for each observation. Only the observations with at least a 
        coverage of `thr` across the timepoints are kept.
        - `n` -> number of random observations to draw from the dataset.
        - `min_frac` -> percentage of the cumulative distribution that will be used as a threshold 
        for each timepoint to remove observations with a too low coverage. 
        It will perform a K-means, for `k` in `k_interval`, to search for the best `K` (optimizing 
        the input metric), and performing K-means to discard observations belonging to clusters 
        with the centroid of all timepoints below `5%` of the sum of centroids for all the timepoints.
        - `k_interval` -> interval of `K` values to look for the best `K`.
        - `metric` -> metric used to retrieve the best `K`, among `calinski_harabasz_score` and `silhouette`.
        - `seed` -> seed value used to compute the Kmeans to perform the initial clustering to filter the data.
        '''
        idxs = torch.any(self.dataset >= min_cov, dim=1)
        try: self.IS = self.IS[idxs]
        except: pass
        finally: self.dataset = self.dataset[idxs,]

        self.dataset, self.IS = self._filter_dataframe_init(min_frac=min_frac,
            k_interval=k_interval, metric=metric, seed=seed)

        if n is not None:  # takes a random sample of `n` elements from the dataset
            n = min(n, self.dataset.shape[0])
            np.random.seed(seed)
            idx = np.random.randint(self.dataset.shape[0], size=n)
            try: self.IS = self.IS[idx]
            finally: self.dataset = self.dataset[idx,:]

        self._N = self.dataset.shape[0]
        self.params["N"] = self._N
        self.init_params["N"] = self._N
        self._initialize_attributes()


    def _initialize_sigma_constraint(self):
        '''
        Function to initialize the constraints on the variance for each dimension in each cluster. 
        It performs a linear regression on the marginal distribution of each dimension and performs 
        a check on the x-intercept, to avoid negative values of y for x in [0,max_cov].
        '''
        if self._default_lm:
            return self._set_sigma_constraints()
        return self._compute_sigma_constraints()


    def _set_sigma_constraints(self):
        slope = self.hyperparameters["slope"]
        intercept = self.hyperparameters["intercept"]
        if isinstance(slope, torch.Tensor) and isinstance(intercept, torch.Tensor):
            return {"slope":slope, "intercept":intercept}

        slope_tns = torch.repeat_interleave(slope, self._T)
        intercept_tns = torch.repeat_interleave(intercept, self._T)
        lm = {"slope":slope_tns, "intercept":intercept_tns}
        
        self.hyperparameters["slope"] = slope_tns
        self.hyperparameters["intercept"] = intercept_tns
        return lm


    def _compute_sigma_constraints(self):
        lm = dict()
        slope, intercept = torch.zeros((self._T)), torch.zeros((self._T))
        for t in range(self._T):
            xx, yy = np.unique(self.dataset[:,t], return_counts=True)
            lmodel = sklearn.linear_model.LinearRegression()
            fitted = lmodel.fit(xx.reshape(-1,1), yy.reshape(-1,1))  # eatimate the coefficient of the linear reg
            slope[t] = torch.tensor(float(fitted.coef_[0]))
            intercept[t] = torch.tensor(fitted.intercept_[0])
            
            # check that y=0 when x >= (max_cov*1.5 + max_cov/10)
            if (slope[t] * torch.tensor(max(xx)*1.5) + intercept[t]) <= (max(xx) / 10):
                intercept[t] = torch.max(-1*( slope[t] * torch.tensor(max(xx)*1.5) ), \
                    torch.tensor(max(xx) / 10))
        lm["slope"], lm["intercept"] = slope, intercept

        self.hyperparameters["slope"] = slope
        self.hyperparameters["intercept"] = intercept
        return lm


    def _filter_dataframe_init(self, min_frac, k_interval, seed, \
            metric="calinski_harabasz_score"):
        '''
        Function to filter the input dataset according to the centroid the clusters output
        from a KMeans, with `K` being the best `K` in `k_interval` according to `metric`.
        - `min_frac` -> percentage of the cumulative distribution that will be used as a threshold 
        for each timepoint to remove observations with a too low coverage. 
        It will perform a K-means, for `k` in `k_interval`, to search for the best `K` (optimizing 
        the input metric), and performing K-means to discard observations belonging to clusters 
        with the centroid of all timepoints below `5%` of the sum of centroids for all the timepoints.
        - `k_interval` -> interval of `K` values to look for the best `K`.
        - `metric` -> metric used to retrieve the best `K`, among `calinski_harabasz_score` and `silhouette`.
        - `seed` -> seed used for the KMeans.
        '''

        index_fn = self._find_index_function(metric)
        N, K = self.dataset.shape[0], self._find_best_k(k_interval=k_interval, index_fn=index_fn, seed=seed)
        km = self.run_kmeans(K, seed=seed)

        clusters = km.labels_
        ctrs = torch.tensor(km.cluster_centers_).float().detach() + torch.abs(torch.normal(0, 1, (K, self._T)))
        keep = torch.where((ctrs / ctrs.sum(dim=0) > min_frac).sum(dim=1) > 0)[0]

        try: ii = self.IS[np.in1d(np.array(clusters), keep)]
        except: ii = self.IS
        finally: return self.dataset[np.in1d(np.array(clusters), keep)], ii


    def _find_index_function(self, index="calinski_harabasz_score"):
        if index == "calinski_harabasz_score":
            return sklearn.metrics.calinski_harabasz_score
        if index == "silhouette":
            return sklearn.metrics.silhouette_score


    def _find_best_k(self, k_interval, seed, index_fn=sklearn.metrics.calinski_harabasz_score):
        k_min = min(max(k_interval[0], 2), self.dataset.unique(dim=0).shape[0]-1)
        k_max = min(k_interval[1], self.dataset.unique(dim=0).shape[0]-1)

        if k_min > k_max:
            k_max = k_min + 1
        if k_min == k_max:
            return k_min

        k_interval = (k_min, k_max)

        scores = torch.zeros(k_interval[1])
        for k in range(k_interval[0], k_interval[1]):
            km = self.run_kmeans(k, seed=seed)
            labels = km.labels_
            real_k = len(np.unique(labels))
            scores[real_k] = max(scores[real_k], index_fn(self.dataset, labels))

        best_k = scores.argmax()  # best k is the one maximing the calinski score
        return best_k


    def run_kmeans(self, K, seed):
        removed_idx, data_unq = self.check_input_kmeans()

        km = KMeans(n_clusters=K, random_state=seed).fit(data_unq.numpy())
        assert km.n_iter_ < km.max_iter

        clusters = km.labels_
        for rm in sorted(removed_idx.keys()):
            # insert 0 elements to restore the original number of obs
            clusters = np.insert(clusters, rm, 0, 0)

        for rm in removed_idx.keys():
            # insert in the repeated elements the correct cluster
            rpt = removed_idx[rm]  # the index of the kept row
            clusters[rm] = clusters[rpt]

        km.labels_ = clusters
        return km


    def check_input_kmeans(self):
        '''
        Function to check the inputs of the Kmeans. There might be a problem when multiple observations 
        are equal since the Kmeans will keep only a unique copy of each and the others will not be initialized.
        '''
        a = self.dataset.numpy()

        tmp, indexes, count = np.unique(a, axis=0, return_counts=True, return_index=True)
        repeated_groups = tmp[count > 1].tolist()

        unq = torch.from_numpy(np.array([a[index] for index in sorted(indexes)]))

        removed_idx = {}
        for i, repeated_group in enumerate(repeated_groups):
            rpt_idxs = np.argwhere(np.all(a == repeated_group, axis=1)).flatten()
            removed = rpt_idxs[1:]
            for rm in removed:
                removed_idx[rm] = rpt_idxs[0]

        return removed_idx, unq


    def model(self):
        N, K = self._N, self.K

        weights = pyro.sample("weights", distr.Dirichlet(torch.ones(K)))  # mixing proportions for each component sample the mixing proportion

        mean_scale = self.hyperparameters["mean_scale"]
        mean_loc = self.hyperparameters["mean_loc"]
        var_loc = self.hyperparameters["var_loc"]
        var_scale = self.hyperparameters["var_scale"]
        eta = self.hyperparameters["eta"]
        var_constr = self.init_params["var_constr"]

        with pyro.plate("time_plate", self._T):
            with pyro.plate("comp_plate", K):
                mean = pyro.sample("mean", distr.Normal(mean_loc, mean_scale))

        with pyro.plate("time_plate2", self._T):
            with pyro.plate("comp_plate3", K):
                variant_constr = pyro.sample("var_constr", distr.Delta(var_constr))
                sigma_vector = pyro.sample("sigma_vector", distr.Normal(var_loc, var_scale))  # sampling sigma, the sd

        if self.cov_type == "diag" or self._T == 1:
            sigma_chol = torch.eye(self._T) * 1.
        if self.cov_type == "full" and self._T > 1:
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
                params = self._initialize_params()
                N, K = params["N"], params["K"]
                min_var = self.hyperparameters["min_var"]

                weights_param = pyro.param("weights_param", lambda: params["weights"], \
                    constraint=constraints.simplex)
                mean_param = pyro.param("mean_param", lambda: params["mean"], \
                    constraint=constraints.positive)

                if self.cov_type=="full" and self._T > 1:
                    with pyro.plate("comp_plate2", K):
                        sigma_chol_param = pyro.param("sigma_chol_param", lambda: params["sigma_chol"], \
                            constraint=constraints.corr_cholesky)
                        # sigma_chol_param has shape [10,12,12]
                        # to_event(2) takes the two rightmost dimensions ([12,12]) and the last 
                        # dimension [10] is now i.i.d. 
                        # -> so we have 10 elements of shape [12,12] sampled independently
                        sigma_chol = pyro.sample("sigma_chol", distr.Delta(sigma_chol_param).to_event(2))
                
                elif self.cov_type=="diag" or self._T == 1:
                    sigma_chol_param = pyro.param("sigma_chol_param", lambda: params["sigma_chol"])
                
                # to_event(1) makes the elements sampled independently
                weights = pyro.sample("weights", distr.Delta(weights_param).to_event(1))
                with pyro.plate("time_plate", self._T):
                    with pyro.plate("comp_plate", K):
                        mean = pyro.sample("mean", distr.Delta(mean_param))

                with pyro.plate("time_plate2", self._T):
                    with pyro.plate("comp_plate3", K):
                        variant_constr = pyro.sample(f"var_constr", distr.Delta(params["var_constr"]))
                        sigma_vector_param = pyro.param(f"sigma_vector_param", lambda: params["sigma_vector"], 
                            constraint=constraints.interval(min_var, variant_constr))
                        sigma_vector = pyro.sample(f"sigma_vector", distr.Delta(sigma_vector_param))

                with pyro.plate("data_plate", N):
                    z = pyro.sample("z", distr.Categorical(weights), \
                        infer={"enumerate":self._enumer})
            return guide_expl
        else:
            raise NotImplementedError


    def compute_Sigma(self, sigma_chol, sigma_vector, K):
        '''
        Function to compute the sigma_tril used in the Normal likelihood
        '''
        Sigma = torch.zeros((K, self._T, self._T))
        for k in range(K):
            if self.cov_type == "diag" or self._T == 1:
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].diag_embed(), \
                    sigma_chol).add(torch.eye(self._T))
            if self.cov_type == "full" and self._T > 1:
                Sigma[k,:,:] = torch.mm(sigma_vector[k,:].diag_embed(), \
                    sigma_chol[k]).add(torch.eye(self._T))
        return Sigma


    def _initialize_params(self):
        '''
        Function to initialize the parameters.
        It performs a K-means clustering and define the initial weights 
        as the proportion of IS assigned to each cluster and the
        Poisson lambda or Gaussian mean as the centers of each cluster.
        '''
        if not self.init_params["is_computed"]:
            N, K = self.params["N"], self.params["K"]

            ctrs = self._initialize_centroids(K, N)
            var, var_constr = self._initialize_variance(K, ctrs)
            sigma_chol = self._initialize_sigma_chol(K)
            Sigma = self.compute_Sigma(sigma_chol=sigma_chol, sigma_vector=var, K=K)

            self.init_params["mean"] = ctrs
            self.init_params["sigma_vector"] = var
            self.init_params["var_constr"] = var_constr
            self.init_params["sigma_chol"] = sigma_chol
            self.init_params["sigma"] = Sigma

            self.init_params["is_computed"] = True

        return self.init_params


    def _initialize_kmeans(self, K, seed):
        '''
        Function to find the optimal seed for the initial KMeans, checking the inertia.
        '''

        km = self.run_kmeans(K, seed=seed)
        # score = sklearn.metrics.calinski_harabasz_score(self.dataset, km.labels_)
        # return score, seed
        return np.round(km.inertia_, 3), seed


    def _initialize_centroids(self, K, N):
        if self._init_seed is None:
            _, self._init_seed = min([self._initialize_kmeans(K, seed) for seed in range(10)], key=lambda x: x[0])

        km = self.run_kmeans(K, seed=self._init_seed)
        self.init_params["clusters"] = torch.from_numpy(km.labels_)

        # init the mixing proportions
        w = torch.tensor([(np.where(km.labels_ == k)[0].shape[0]) / N for k in range(km.n_clusters)])
        self.init_params["weights"] = w.float().detach()

        # add gaussian noise to the centroids and reset too low values
        ctrs = torch.tensor(km.cluster_centers_).float().detach() + \
            torch.abs(torch.normal(0, 1, (self.K, self._T)))

        ctrs[ctrs <= 0] = 0.01
        return ctrs


    def _initialize_variance(self, K, ctrs):
        # set the linear model for the variance constraints
        self.lm = self._initialize_sigma_constraint()

        var = torch.zeros(self.params["K"], self._T)
        var_constr = torch.zeros(self.params["K"], self._T)

        for cl in torch.unique(self.init_params["clusters"]):
            var[cl,:] = torch.var( self.dataset[ \
                torch.where(self.init_params["clusters"]==cl) ].float(), \
                dim=0, unbiased=False)
            var_constr[cl,:] = ctrs[cl,:] * self.lm["slope"] + self.lm["intercept"]
            # var_constr[cl,:] = var_constr[cl,:] * 1.5

        # add gaussian noise to the variance
        var += torch.abs(torch.normal(0, 1, (K, self._T)))

        # reset variance contraints and variance values
        max_var = self.hyperparameters.get("max_var", None)
        min_var = self.hyperparameters.get("min_var", None)
        if max_var is not None:
            # reset if values are larger than the maximum value set
            var_constr[var_constr > max_var] = max_var - .1
        if min_var is not None:
            # reset if values are lower than the minimum value set
            var_constr[var_constr < min_var] = min_var + .1

        # reset variance values if larger than the variance constraint
        var[var > var_constr] = var_constr[var > var_constr] - .1
        var[var <= 0] = 1.

        return var, var_constr


    def _initialize_sigma_chol(self, K):
        if self.cov_type == "diag" or self._T == 1:
            sigma_chol = torch.eye(self._T) * 1.
        
        elif self.cov_type == "full" and  self._T > 1:
            sigma_chol = torch.zeros((K, self._T, self._T))
            for cl in range(K):
                sigma_chol[cl,:] = distr.LKJCholesky(self._T, self.hyperparameters["eta"]).sample()

            # sigma_chol = torch.zeros((K, self._T, self._T))
            # for cl in range(K):

            #     # filter only the cluster's observations
            #     data = self.dataset[ \
            #         torch.where(self.init_params["clusters"]==cl) ].float()

            #     cov = torch.cov(data.t())
            #     corr = torch.zeros_like(cov)

            #     if data.shape[0] == 1:
            #         print("INSIDE IF")
            #         sigma_chol[cl,:] = torch.cholesky(torch.eye(self._T), upper=False)
            #         continue

            #     for t1 in range(self._T):
            #         for t2 in range(self._T):
            #             denom = torch.max(torch.tensor(1e-5), sd[cl,t1] * sd[cl,t2])
            #             corr[t1,t2] = cov[t1,t2] / denom

            #     print(corr[10:,10:])

            #     sigma_chol[cl,:] = torch.linalg.cholesky(torch.mm(corr, corr.t()).add(self._T))

        return sigma_chol


    def _initialize_seed(self, optim, elbo, seed):
        '''
        Function used to optimize the initialization of the SVI object.
        '''
        pyro.set_rng_seed(seed)
        pyro.get_param_store().clear()

        guide = self.guide()
        svi = SVI(self.model, guide, optim, elbo) # reset Adam params each seed

        loss = svi.step()

        self.init_params["is_computed"] = False
        return np.round(loss, 3), seed


    def fit(self, steps=500, optim_fn=pyro.optim.Adam, loss_fn=pyro.infer.TraceEnum_ELBO(), \
            lr=0.005, cov_type="full", check_conv=True, p=1,  min_steps=20, default_lm=True, \
            store_params=False, show_progr=True, seed_optim=True, seed=5, init_seed=None):
        
        pyro.enable_validation(True)

        self._settings = {"optim":optim_fn({"lr":lr}), "loss":loss_fn, "lr":lr}
        self._is_trained = False
        self._max_iter = steps
        self.cov_type = cov_type
        self.init_params["is_computed"] = False
        self._default_lm = default_lm
        self._seed = None if seed_optim else seed
        self._init_seed = None if init_seed is None else init_seed

        if seed_optim:
            # set the guide and initialize the SVI object to minimize the initial loss 
            # for each seed in the range, it will initialize a global guide with the initialized parameters
            optim = self._settings["optim"]
            elbo = self._settings["loss"]
            _, self._seed = min([self._initialize_seed(optim, elbo, seed) for seed in range(50)], key = lambda x: x[0])
            # print("BEST SEED ", self._seed)
        
        if self._seed is not None:
            pyro.set_rng_seed(self._seed)

        pyro.get_param_store().clear()
        # set the guide and intialize the SVI object
        self._global_guide = self.guide()
        self.svi = SVI(self.model, self._global_guide, \
                    optim=self._settings["optim"], loss=self._settings["loss"])
        self.svi.step()

        losses_grad = self._train(steps=self._max_iter, \
            check_conv=check_conv, p=p, min_steps=min_steps, \
            store_params=store_params, show_progr=show_progr)

        self._is_trained = True
        self.losses_grad_train = losses_grad  # store the losses and the gradients for weights/lambd
        self.guide_trained = self._global_guide  # store the trained guide and model
        self.model_trained = self.model
        self.params = self._get_learned_parameters()
        return


    def _train(self, steps, check_conv, min_steps, p, show_progr, store_params):
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
        params = dict()
        mean_conv, sigma_conv = [self.init_params["mean"],self.init_params["mean"]], \
            [self.init_params["sigma_vector"],self.init_params["sigma_vector"]]  #to store the lambda at the previous and current iteration
        conv = 0

        t = trange(steps, desc='Bar desc', leave=True) if show_progr else range(steps)
        for step in t:
            self.iter = step
            elb = self.svi.step() / self._N
            losses.append(elb)

            params_step = self._get_learned_parameters() 

            if store_params:
                for k,v in params_step.items():
                    if k in ["weights", "mean", "sigma_vector", "sigma_chol"]:
                        params[k] = params.get(k, dict())
                        params[k]["step_"+str(step)] = v.clone().detach().numpy()

            # gradient_norms = self._reset_params(params=params_step, p=.01, gradient_norms=gradient_norms)
            if check_conv and step >= min_steps:
                mean_conv[0], mean_conv[1] = mean_conv[1], params_step["mean"]
                sigma_conv[0], sigma_conv[1] = sigma_conv[1], params_step["sigma_vector"]
                losses_conv = losses[-2:]
                conv = self._convergence(mean_conv, sigma_conv, losses_conv, conv, p=p)

                if conv == 10:
                    if show_progr:
                        t.set_description("ELBO %f" % elb)
                        t.reset(total=step)
                    break

            if show_progr:
                t.set_description("ELBO %f" % elb)
                t.refresh()
        
        return {"losses":losses, 
            "gradients":dict(gradient_norms), 
            "params":params}


    def _convergence(self, mean_conv, sigma_conv, elbo, conv, p):
        perc = p * self._settings["lr"]
        if self._convergence_elbo(elbo) and \
            self._check_convergence(mean_conv, perc) and \
            self._check_convergence(sigma_conv, perc):
            return conv + 1
        return 0


    def _convergence_elbo(self, elbo, p=.05):
        '''
        Function to check the convergence of the ELBO.
        - `elbo` is a list with the ELBO at iteration `t` and `t-1`.
        - `p` is the percentage of ELBO s.t. `| E^(t-1) - E^(t) | < p*E^(t-1)`
        '''
        return True
        eps = p * elbo[0]
        return abs(elbo[0] - elbo[1]) <= eps


    # def _convergence_grads(self, gradient_norms, conv):
    #     for gr in gradient_norms.keys():
    #         if gr in ["mean_param", "sigma_vector_param"]:
    #             prev = gradient_norms[gr][-2]
    #             curr = gradient_norms[gr][-1]
    #             cc = np.abs(prev - curr) < curr*p
    #     if cc:
    #         return conv +1
    #     return 0



    def _normalize(self, par):
        # zi = (xi - min(x)) / (max(x) - min(x))
        norm = list()
        for p in par:
            if torch.min(p) == torch.max(p):
                norm.append(p)
            norm.append(( p - torch.min(p) ) / ( torch.max(p) - torch.min(p) ) * 100)
        return norm


    def _check_convergence(self, par, perc) -> Boolean:
        '''
        - `par` -> list of 2 elements given the previous (at index 0) and current (at index 1) 
        estimated values for a parameter.
        - `perc` -> numeric value in `[0,1]`, corresponding to a percentage used for convergence.

        The function returns `True` if all the values changed less than `perc*100`% 
        in the current step with respect to the previous one.
        '''
        par = self._normalize(par)
        n = 0
        for k in range(par[0].shape[0]):
            for t in range(par[0].shape[1]):
                eps = perc * par[0][k,t] # torch.max(torch.tensor(1), par[0][k,t])
                if torch.absolute(par[0][k,t] - par[1][k,t]) <= eps:
                    n += 1

        return n >= 0.9 * self.params["K"]*self.params["T"]


    def _get_learned_parameters(self) -> Dict:
        '''
        Function to store and return the MAP estimate of the 
        parameters from the `pyro.param_store()`. \\
        It returns a dictionary 
        '''
        param_store = pyro.get_param_store()
        if self._autoguide:
            raise NotImplementedError

        p = {}
        p["N"], p["K"], p["T"] = self.params["N"], self.params["K"], self.params["T"]
        p["weights"] = param_store["weights_param"].clone().detach()

        p["mean"] = param_store["mean_param"].clone().detach()
        p["sigma_vector"] = param_store["sigma_vector_param"].clone().detach()
        p["sigma_chol"] = param_store["sigma_chol_param"].clone().detach()

        if self._is_trained:
            p["sigma"] = self.compute_Sigma(p["sigma_chol"], p["sigma_vector"], p["K"])
        return p


    def classifier(self, params=None, perc=.5, t=10):
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

        n = min(np.ceil(p/100 * params["N"]), t)
        keep_low = torch.tensor(np.where( (params["z_assignments"].sum(dim=0) <= n) & \
            (params["z_assignments"].sum(dim=0) > 0) )[0])

        for cluster in keep_low:
            # count = 0
            obs = torch.tensor(np.where(params["z_assignments"][:,cluster]==1)[0])
            for o in obs:
                if params["z_probs"][o, cluster] < min_prob:
                    keep_low = torch.cat([keep_low[0:cluster], keep_low[cluster+1:]])
                    break

        keep = torch.tensor(np.where(params["z_assignments"].sum(dim=0) > n)[0])
        try: keep = torch.unique(torch.cat([keep, keep_low]))
        except: keep = keep
        
        if len(keep) == 0:
            return params
        
        params["K"] = len(keep)
        params.pop("z_assignments")
        params.pop("z_probs")
        for par in params.keys():
            if self.cov_type=="full" and par in ["N", "K", "T", None]:
                continue
            if (self.cov_type=="diag" or self._T == 1) and par in ["N", "K", "T", "sigma_chol", None]:
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
            # return self._input_K + \
            #     self._input_K*self._T + \
            #     self._input_K*self._T + \
            #     self._input_K * self._T*(self._T-1) / 2
            return params["weights"].numel() + params["mean"].numel() \
                + params["sigma_vector"].numel() + params["K"]*self._T*(self._T-1) / 2

        if self.cov_type == "diag":
            # return self._input_K + \
            #     self._input_K*self._T + \
            #     self._input_K*self._T
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
