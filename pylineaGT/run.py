import pandas as pd
import pyro
from random import randint
from .mvnmm import MVNMixtureModel

def run_inference(cov_df, IS=[], columns=[], lineages=[], k_interval=[10,30], 
        n_runs=1, steps=500, lr=0.005, p=1, convergence=True, initializ=False,
        covariance="full", hyperparameters=dict(), show_progr=True, 
        store_grads=True, store_losses=True, store_params=True, seed=5, init_seed=None):

    if init_seed is None:  # init seed is the seed for the params initialization -> specific for each run
        init_seed = [randint(1,5) for _ in range(n_runs)]
        while len(set(init_seed)) != len(init_seed):
            init_seed = [randint(1,10) for _ in range(n_runs)]
    
    ic_df = pd.DataFrame(columns=["K","run","NLL","BIC","AIC","ICL"])
    
    losses_df = pd.DataFrame(columns=["K","run","losses"])
    losses_df.losses = losses_df.losses.astype("object")
    
    grads_df = pd.DataFrame(columns=["K","run","param","grad_norm"])
    grads_df.grad_norm = grads_df.grad_norm.astype("object")
    
    params_df = pd.DataFrame(columns=["K","run","param","params_values"])
    params_df.params_values = params_df.params_values.astype("object")

    for k in range(k_interval[0], k_interval[1]+1):
        for run in range(1, n_runs+1):
            # at the end of each run I would like:
            # - losses of the run
            # - AIC/BIC/ICL
            # - gradient norms for the parameters
            print(cov_df.shape)
            x_k = single_run(k=k, df=cov_df, IS=IS, columns=columns, lineages=lineages, 
                steps=steps, covariance=covariance, lr=lr, p=p, initializ=initializ,
                hyperparameters=hyperparameters, convergence=convergence, 
                show_progr=show_progr, store_params=store_params, 
                seed=seed, init_seed=init_seed[run-1])
            print(cov_df.shape)

            kk = x_k.params["K"]

            # print(x_k._noise)
            # print(kk)
            # print(x_k.compute_ic(method="BIC"))
            # print(init_seed[run-1], x_k._seed)

            best_init_seed = init_seed[run-1]
            best_seed = x_k._seed

            id = '.'.join([str(k), str(run)])

            if store_grads: grads_df = pd.concat([grads_df, compute_grads(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)
            if store_losses: losses_df = pd.concat([losses_df, compute_loss(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list
            if store_params: params_df =  pd.concat([params_df, retrieve_params(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list
            
            ic_df = pd.concat([ic_df, compute_ic(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)

    return ic_df, losses_df, grads_df, params_df


def single_run(k, df, IS=[], columns=[], lineages=[], steps=500, covariance="full", 
    lr=0.005, p=1, convergence=True, initializ=False, show_progr=True, 
    hyperparameters=dict(), store_params=False, seed=5, init_seed=1):

    pyro.clear_param_store()
    try:
        columns = df.columns[df.columns.str.startswith("cov")].to_list()
        IS = df.IS.to_list()
        x = MVNMixtureModel(k, data=df[columns], lineages=lineages, IS=IS, columns=columns)
    except:
        IS = ["IS.".join(str(i)) for i in range(df.shape[0])]
        x = MVNMixtureModel(k, data=df, lineages=lineages, IS=IS)

    for name, value in hyperparameters.items():
        x.set_hyperparameters(name, value)

    x.fit(steps=steps, cov_type=covariance, lr=lr, p=p, convergence=convergence, 
        show_progr=show_progr, store_params=store_params, initializ=initializ, 
        seed=seed, init_seed=init_seed)
    x.classifier()

    return x


def compute_grads(model, kk, run, id, init_seed, seed):
    return pd.DataFrame({"K":kk, 
        "run":run, 
        "id":id,
        "seed":seed,
        "init_seed":init_seed,
        "param":["mean_param","sigma_vector_param","weights_param"],
        "grad_norm":[model.losses_grad_train["gradients"]["mean_param"],
                     model.losses_grad_train["gradients"]["sigma_vector_param"],
                     model.losses_grad_train["gradients"]["weights_param"]]})


def compute_loss(model, kk, run, id, init_seed, seed):
    return pd.DataFrame({"K":kk, 
        "id":id, 
        "run":run, 
        "seed":seed, 
        "init_seed":init_seed,
        "losses":[model.losses_grad_train["losses"]]})


def retrieve_params(model, kk, run, id, init_seed, seed):
    return pd.DataFrame({"K":kk, 
        "run":run, 
        "id":id,
        "seed":seed,
        "init_seed":init_seed,
        "param":["mean","sigma_vector","weights"],
        "params_values":[model.losses_grad_train["params"]["mean"],
                         model.losses_grad_train["params"]["sigma_vector"],
                         model.losses_grad_train["params"]["weights"]]})


def compute_ic(model, kk, run, id, init_seed, seed):
    ic_dict = {"K":[kk], "run":[run], "id":[id], "seed":[seed], "init_seed":[init_seed]}
    ic_dict["NLL"] = [float(model.compute_ic(method="NLL"))]
    ic_dict["BIC"] = [float(model.compute_ic(method="BIC"))]
    ic_dict["AIC"] = [float(model.compute_ic(method="AIC"))]
    ic_dict["ICL"] = [float(model.compute_ic(method="ICL"))]
    return pd.DataFrame(ic_dict)
