import pandas as pd
from .mvnmm import MVNMixtureModel

def run_inference(cov_df, lineages, k_interval=[10,30], n_runs=2, steps=500, lr=0.005,
        p=0.01, convergence=True, covariance="diag", show_progr=True, 
        store_grads=True, store_losses=True, random_state=25):

    ic_df = pd.DataFrame(columns=["K","run","NLL","BIC","AIC","ICL"])
    losses_df = pd.DataFrame(columns=["K","run","losses"])
    losses_df.losses = losses_df.losses.astype("object")
    grads_df = pd.DataFrame(columns=["K","run","param","grad_norm"])
    grads_df.grad_norm = grads_df.grad_norm.astype("object")

    for k in range(k_interval[0], k_interval[1]+1):
        for run in range(1, n_runs+1):
            # at the end of each run I would like:
            # - losses of the run
            # - AIC/BIC/ICL
            # - gradient norms for the parameters
            x_k = single_run(k, cov_df, lineages, run, steps, covariance, lr, p, convergence, show_progr, random_state)

            kk = x_k.params["K"]

            if store_grads: grads_df = pd.concat([grads_df, compute_grads(x_k, kk, run)], ignore_index=True)
            if store_losses: losses_df = pd.concat([losses_df, compute_loss(x_k, kk, run)], ignore_index=True)  # list
            
            ic_df = pd.concat([ic_df, compute_ic(x_k, kk, run)], ignore_index=True)

    return ic_df, losses_df, grads_df


def single_run(k, df, lineages, run="", steps=500, covariance="diag", lr=0.001,
    p=0.01, convergence=True, show_progr=True, random_state=25):

    try:
        columns = df.columns[df.columns.str.startswith("cov")].to_list()
        IS = df.IS.to_list()
        x = MVNMixtureModel(k, data=df[columns], lineages=lineages, IS=IS, columns=columns)
    except:
        IS = ["IS.".join(str(i)) for i in range(df.shape[0])]
        x = MVNMixtureModel(k, data=df, lineages=lineages, IS=IS)
    
    x.fit(steps=steps, cov_type=covariance, lr=lr, p=p, convergence=convergence, random_state=random_state, show_progr=show_progr)
    x.classifier()

    return x


def compute_grads(model, kk, run):
    return pd.DataFrame({"K":kk, 
        "run":run, 
        "param":["mean_param","sigma_vector_param","weights_param"],
        "grad_norm":[model.losses_grad_train["gradients"]["mean_param"],
                     model.losses_grad_train["gradients"]["sigma_vector_param"],
                     model.losses_grad_train["gradients"]["weights_param"]]})


def compute_loss(model, kk, run):
    return pd.DataFrame({"K":kk, "run":run, "losses":[model.losses_grad_train["losses"]]})


def compute_ic(model, kk, run):
    ic_dict = {"K":[kk], "run":[run]}
    ic_dict["NLL"] = [float(model.compute_ic(method="NLL"))]
    ic_dict["BIC"] = [float(model.compute_ic(method="BIC"))]
    ic_dict["AIC"] = [float(model.compute_ic(method="AIC"))]
    ic_dict["ICL"] = [float(model.compute_ic(method="ICL"))]
    return pd.DataFrame(ic_dict)
