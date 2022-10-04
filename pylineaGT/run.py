import pandas as pd
import pyro

from .mvnmm import MVNMixtureModel

def run_inference(cov_df, IS=[], columns=[], lineages=[], k_interval=[10,30], 
        n_runs=1, steps=500, lr=0.005, p=1, check_conv=True, min_steps=20,
        covariance="full", hyperparams=dict(), default_lm=True, show_progr=True, 
        store_grads=True, store_losses=True, store_params=True, seed_optim=True, 
        seed=5, init_seed=None):

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
            x_k = single_run(k=k, df=cov_df, IS=IS, columns=columns, lineages=lineages, 
                steps=steps, covariance=covariance, lr=lr, p=p, check_conv=check_conv, 
                min_steps=min_steps, default_lm=default_lm, hyperparams=hyperparams, 
                show_progr=show_progr, store_params=store_params, 
                seed_optim=seed_optim, seed=seed, init_seed=init_seed)

            if x_k == 0:
                continue

            kk = x_k.params["K"]

            best_init_seed = x_k._init_seed
            best_seed = x_k._seed

            id = '.'.join([str(k), str(run)])

            if store_grads: grads_df = pd.concat([grads_df, compute_grads(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)
            if store_losses: losses_df = pd.concat([losses_df, compute_loss(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list
            if store_params: params_df =  pd.concat([params_df, retrieve_params(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)  # list
            
            ic_df = pd.concat([ic_df, compute_ic(x_k, kk, run, id, best_init_seed, best_seed)], ignore_index=True)
    
    return ic_df, losses_df, grads_df, params_df


def single_run(k, df, IS, columns, lineages, steps, covariance, lr, check_conv, min_steps, p, 
    hyperparams, default_lm, show_progr, store_params, seed_optim, seed, init_seed):

    pyro.clear_param_store()
    try:
        columns = df.columns[df.columns.str.startswith("cov")].to_list()
        IS = df.IS.to_list()
        x = MVNMixtureModel(k, data=df[columns], lineages=lineages, IS=IS, columns=columns)
    except:
        IS = ["IS.".join(str(i)) for i in range(df.shape[0])]
        x = MVNMixtureModel(k, data=df, lineages=lineages, IS=IS)

    if x.error:
        return 0

    for name, value in hyperparams.items():
        x.set_hyperparameters(name, value)

    x.fit(steps=steps, cov_type=covariance, lr=lr, check_conv=check_conv, p=p,
        min_steps=min_steps, default_lm=default_lm, show_progr=show_progr, 
        store_params=store_params, seed_optim=seed_optim, seed=seed, init_seed=init_seed)

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

    # print(kk, ic_dict["BIC"])

    return pd.DataFrame(ic_dict)


