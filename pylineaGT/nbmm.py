import torch
import pyro
import pyro.distributions as distr
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, autoguide
from pyro.optim import Adam
from torch.distributions import constraints
from tqdm import trange


def model(X, K):
    N, T = X.shape

    weights = pyro.sample("weights", distr.Dirichlet(torch.ones(K)))

    with pyro.plate("time_plate", T):
        with pyro.plate("comp_plate", K):
            # number of successes
            alpha = pyro.sample("alpha", distr.Gamma(2.0, 0.5))
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


def fit_mixture(X, K, lr=0.01, steps=1000):
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})

    guide_auto = autoguide.AutoDelta(poutine.block(model, hide=["z"]))
    svi = SVI(model, guide_auto, optimizer, loss=TraceEnum_ELBO())

    t = trange(steps, desc='Bar desc', leave=True)
    for step in t:
        elb = svi.step(X, K)
        if step % 500 == 0:
            t.set_description("ELBO %f" % elb)
            t.refresh()


    return {name: pyro.param(name).detach() for name in pyro.get_param_store().get_all_param_names()}


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


# def run_inference(cov_df, IS=[], columns=[], lineages=[], k_interval=[10,30], 
#         n_runs=1, steps=500, lr=0.005, p=1, check_conv=True, min_steps=20,
#         covariance="full", hyperparams=dict(), default_lm=True, show_progr=True, 
#         store_grads=True, store_losses=True, store_params=True, seed_optim=True, 
#         seed=5, init_seed=None, return_object=False):

#     ic_df = pd.DataFrame(columns=["K","run","NLL","BIC","AIC","ICL"])

#     losses_df = pd.DataFrame(columns=["K","run","losses"])
#     losses_df.losses = losses_df.losses.astype("object")

#     grads_df = pd.DataFrame(columns=["K","run","param","grad_norm"])
#     grads_df.grad_norm = grads_df.grad_norm.astype("object")

#     params_df = pd.DataFrame(columns=["K","run","param","params_values"])
#     params_df.params_values = params_df.params_values.astype("object")

#     for k in range(k_interval[0], k_interval[1]+1):
#         for run in range(1, n_runs+1):
#             # at the end of each run I would like:
#             # - losses of the run
#             # - AIC/BIC/ICL
#             # - gradient norms for the parameters
#             x_k = single_run(k=k, df=cov_df, IS=IS, columns=columns, lineages=lineages, 
#                 steps=steps, covariance=covariance, lr=lr, p=p, check_conv=check_conv, 
#                 min_steps=min_steps, default_lm=default_lm, hyperparams=hyperparams, 
#                 show_progr=show_progr, store_params=store_params, 
#                 seed_optim=seed_optim, seed=seed, init_seed=init_seed)

#             if x_k == 0:
#                 continue

#             kk = x_k.params["K"]

#             best_init_seed = x_k._init_seed
#             best_seed = x_k._seed

#             id = '.'.join([str(k), str(run)])

#             if store_grads: grads_df = pd.concat([grads_df, compute_grads(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)
#             if store_losses: losses_df = pd.concat([losses_df, compute_loss(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list
#             if store_params: params_df =  pd.concat([params_df, retrieve_params(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list

#             ic_df = pd.concat([ic_df, compute_ic(x_k, k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)
#             best_k = ic_df[ic_df["BIC"] == ic_df["BIC"].min()]["True_K"].values
#             if best_k == k:
#                 x_k.classifier()
#                 best_labels = x_k.params["clusters"]

#     if not return_object:
#         return ic_df, losses_df, grads_df, params_df

#     return ic_df, losses_df, grads_df, params_df, best_labels
