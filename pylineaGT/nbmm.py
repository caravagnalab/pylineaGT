import torch
import pyro
import pyro.distributions as distr
import pyro.poutine as poutine
import pandas as pd
from pyro.infer import SVI, TraceEnum_ELBO, autoguide
from pyro.optim import Adam
from torch.distributions import constraints
from tqdm import trange


def model(X, K):
    N, T = X.shape

    weights = pyro.sample("weights", distr.Dirichlet(torch.ones(K)))

    with pyro.plate("time_plate", T):
        with pyro.plate("comp_plate", K):
            # number of successes
            alpha = pyro.sample("alpha", distr.Gamma(1., 0.01))
            # prob of success
            probs = pyro.sample("probs", distr.Beta(2.0, 2.0))

    with pyro.plate("data_plate", N):
        z = pyro.sample("z", distr.Categorical(weights), infer={"enumerate":"parallel"})
        pyro.sample("obs", distr.NegativeBinomial(alpha[z], probs[z]).to_event(1), obs=X)


def guide(X, K):
    N, T = X.shape

    weights_param = pyro.param("weights_param", lambda: torch.ones(K), constraint=constraints.simplex)

    alpha_param = pyro.param("alpha_param", lambda: torch.rand(K, T) + 1.0, constraint=constraints.positive)
    probs_param = pyro.param("probs_param", lambda: torch.rand(K, T) * 0.5 + 0.25, constraint=constraints.unit_interval)
    pyro.sample("alpha", distr.Delta(alpha_param))
    pyro.sample("probs", distr.Delta(probs_param))

    weights = pyro.sample("weights", distr.Delta(weights_param).to_event(1))
    with pyro.plate("data", N):
        pyro.sample("z", distr.Categorical(weights), infer={"enumerate":"parallel"})


def fit_mixture(X, K, lr=0.01, steps=1000, seed=5):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    optimizer = Adam({"lr": lr})

    guide_auto = autoguide.AutoDelta(poutine.block(model, hide=["z"]))
    svi = SVI(model, guide_auto, optimizer, loss=TraceEnum_ELBO())

    losses = list()
    t = trange(steps, desc='Bar desc', leave=True)
    for step in t:
        elb = svi.step(X, K)
        losses.append(elb)
        if step % 500 == 0:
            t.set_description("ELBO %f" % elb)
            t.refresh()

    params = {name: pyro.param(name).detach() for name in pyro.get_param_store().get_all_param_names()}
    post_probs, assignments = posterior_assignments(X, K, params, model)
    params["K"] = len(assignments.unique())
    params["post_probs"] = post_probs
    params["assignments"] = assignments

    return losses, params


def posterior_assignments(X, K, guide_params, model_func):
    N, T = X.shape

    conditioned_model = poutine.condition(model_func, data=guide_params)
    trace = poutine.trace(conditioned_model).get_trace(X, K)

    z_node = trace.nodes["z"]
    weights = trace.nodes["weights"]["value"]

    alpha = trace.nodes["alpha"]["value"]
    probs = trace.nodes["probs"]["value"]
    
    log_post = torch.zeros(N, K)
    for k in range(K):
        # log p(x_n | z=k) for all values of n
        alpha_k = alpha[k]
        probs_k = probs[k]
        # x = X
        # log pmf NegativeBinomial
        log_p_x = (torch.lgamma(X + alpha_k) - torch.lgamma(alpha_k) - torch.lgamma(X + 1) +
                   alpha_k * torch.log(1 - probs_k) + X * torch.log(probs_k)).sum(dim=1)  # sum over T features
        log_post[:, k] = torch.log(weights[k]) + log_p_x

    post_probs = torch.softmax(log_post, dim=1)
    assignments = torch.argmax(post_probs, dim=1)

    return post_probs, assignments


def compute_bic(X, params):
    weights = params["AutoDelta.weights"]
    alpha = params["AutoDelta.alpha"]
    probs = params["AutoDelta.probs"]

    N, T = X.shape
    K = weights.shape[0]

    log_likelihood = 0.0

    for n in range(N):
        x_n = X[n]
        log_comp = torch.zeros(K)

        for k in range(K):
            alpha_k = alpha[k]
            probs_k = probs[k]

            # log p(x_n | z=k)
            log_px = (torch.lgamma(x_n + alpha_k) - torch.lgamma(alpha_k) - torch.lgamma(x_n + 1)
                      + alpha_k * torch.log(1 - probs_k) + x_n * torch.log(probs_k)).sum()
            log_comp[k] = torch.log(weights[k]) + log_px

        log_likelihood += torch.logsumexp(log_comp, dim=0)

    nll = -log_likelihood.item()
    n_parameters = (K - 1) + K*T + K*T
    bic = 2*nll + n_parameters * torch.log(torch.tensor(N)).item()
    return bic



def run_NB_inference(cov_df, k_interval=[10,30], n_runs=1, steps=500, lr=0.005, 
                     store_losses=True, store_params=True, seed=5, return_object=False):

    ic_df = pd.DataFrame(columns=["True_K","K","run","BIC"])

    losses_df = pd.DataFrame(columns=["True_K","K","run","losses"])
    losses_df.losses = losses_df.losses.astype("object")

    params_df = pd.DataFrame(columns=["True_K","K","run","param","params_values"])
    params_df.params_values = params_df.params_values.astype("object")

    for k in range(k_interval[0], k_interval[1]+1):
        for run in range(1, n_runs+1):
            losses, params = fit_mixture(cov_df, k)

            kk = params["K"]
            best_seed = seed
            id = '.'.join([str(k), str(run)])

            if store_losses: losses_df = pd.concat([losses_df, compute_NB_loss(k, kk, run, id, best_seed, losses)], ignore_index=True)  # list
            if store_params: params_df =  pd.concat([params_df, retrieve_NB_params(k, kk, run, id, best_seed, params)], ignore_index=True)  # list

            ic_df = pd.concat([ic_df, compute_NB_ic(k, kk, run, id, best_seed, cov_df, params)], ignore_index=True)
            best_k = ic_df[ic_df["BIC"] == ic_df["BIC"].min()]["True_K"].values
            if best_k == k:
                best_labels = params["assignments"]

    if not return_object:
        return ic_df, losses_df, params_df

    return ic_df, losses_df, params_df, best_labels


def compute_NB_loss(k, kk, run, id, seed, losses):
    return pd.DataFrame({"True_K":k,
        "K":kk,
        "run":run,
        "id":id,
        "seed":seed,
        "losses":[losses]})


def retrieve_NB_params(k, kk, run, id, seed, params):
    return pd.DataFrame({"True_K":k,
        "K":kk,
        "run":run,
        "id":id,
        "seed":seed,
        "param":["probs","alpha","weights"],
        "params_values":[params["AutoDelta.probs"],
                         params["AutoDelta.alpha"],
                         params["AutoDelta.weights"]]})

def compute_NB_ic(k, kk, run, id, seed, X, params):
    ic_dict = {"True_K":[k], "K":[kk], "run":[run], "id":[id], "seed":[seed]}
    ic_dict["BIC"] = [float(compute_bic(X, params))]

    return pd.DataFrame(ic_dict)

