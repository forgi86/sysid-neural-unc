import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("TKAgg")
import torch.nn as nn
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import wh2009_loader
import matplotlib.pyplot as plt
from diffutils import parameter_jacobian
import functorch


if __name__ == '__main__':

    model_filename = "model.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    #hidden_sizes = model_data["hidden_sizes"]
    #hidden_acts = model_data["hidden_acts"]

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Constants
    n_x = 6
    n_u = 1
    n_y = 1
    seq_est_len = 40
    est_hidden_size = 15
    hidden_size = 15
    idx_start = 0 #5000
    n_fit = 10_000
    n_val = 5_000

    no_cuda = True  # no GPU, CPU only training
    dtype = torch.float32
    threads = 6  # max number of CPU threads
    tau_prior = 0.1  # precision (1/var) of the prior on theta
    sigma_noise = 5e-3  # noise variance (could be learnt instead)

    var_noise = sigma_noise**2
    beta_noise = 1/var_noise
    var_prior = 1 / tau_prior

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    # %% Load dataset
    idx_fit_start = idx_start
    idx_fit_stop = idx_start + n_fit
    t_train, u_train, y_train = wh2009_loader("train", scale=False, dataset_name="WienerHammerSysMs.csv")
    t_fit, u_fit, y_fit = t_train[idx_fit_start:idx_fit_stop], u_train[idx_fit_start:idx_fit_stop], y_train[idx_fit_start:idx_fit_stop]
    N = t_fit.shape[0]

    # Setup neural model structure
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_sizes=[hidden_size], hidden_acts=[nn.Tanh()]).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)

    model.load_state_dict(model_data["model"])
    model = model.to(dtype).to(device)

    n_param = sum(map(torch.numel, model.parameters()))
    n_fparam = sum(map(torch.numel, f_xu.parameters()))
    n_gparam = sum(map(torch.numel, g_x.parameters()))

    u_v = torch.tensor(u_fit[:, None, :], dtype=dtype)
    y_v = torch.tensor(y_fit[:, None, :], dtype=dtype)

    x_0 = torch.zeros((1, n_x), dtype=dtype, device=u_v.device)
    # x_0 = state_estimator(u_val_t, y_val_t)
    y_sim = model(x_0, u_v)

    loss = torch.mean((y_sim - y_v)**2)

    # Exact Hessian computation
    func_model, params = functorch.make_functional(model)

    # Naive jacobian computation
    time_jac_start = time.time()
    #jac_fun = functorch.jacrev(func_model, 0)
    #jacs = jac_fun(params, x_0, u_v)
    #jacs_2d = [jac.reshape(N, -1) for jac in jacs]
    #J = torch.cat(jacs_2d, dim=-1)

    J = parameter_jacobian(model, x_0, u_v, vectorize=True, flatten=True)
    time_jac = time.time() - time_jac_start
    print(f"Jacobian computation time: {time_jac:.2f} s")

    def loss_fun(*model_params):
        y_sim = func_model(model_params, x_0, u_v)
        loss = torch.mean((y_sim - y_v) ** 2)
        return loss

    time_hess_start = time.time()
    H = torch.autograd.functional.hessian(loss_fun, params, vectorize=True)
    time_hess = time.time() - time_hess_start
    print(f"Hessian computation time: {time_hess:.2f} s")

    # Jacobian, experimental
    #time_jac_start = time.time()
    #jac_fun = functorch.jacrev(func_model, 0)
    #jacs = jac_fun(params, x_0, u_v)
    #jacs_2d = [jac.reshape(N, -1) for jac in jacs]
    #JJ = torch.cat(jacs_2d, dim=-1)
    #print(f"Jacobian computation time: {time_jac:.2f} s")
