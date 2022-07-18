import os
import numpy as np
import torch
import matplotlib
matplotlib.use("TKAgg")
import torch.nn as nn
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import wh2009_loader
import matplotlib.pyplot as plt


# Truncated simulation error minimization method
if __name__ == '__main__':

    model_filename = "model.pt"
    model_data = torch.load(os.path.join("models", model_filename))

    cov_filename = "covariance.pt"
    cov_data = torch.load(os.path.join("models", cov_filename))

    P_post = cov_data["P_post"]
    H_post = cov_data["H_post"]
    scaling_H = cov_data["scaling_H"]
    scaling_P = cov_data["scaling_P"]

    # P_post = P_post/scaling_P
    #scaling_phi = cov_data["scaling_phi"]


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
    idx_start = 5000
    n_fit = 170000

    no_cuda = True  # no GPU, CPU only training
    dtype = torch.float64
    threads = 12  # max number of CPU threads
    beta_prior = 0.01  # precision (1/var) of the prior on theta
    sigma_noise = 5e-3  # noise variance (could be learnt instead)

    var_noise = sigma_noise**2
    beta_noise = 1/var_noise
    var_prior = 1/beta_prior

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    # %% Load dataset
    idx_fit_start = idx_start
    idx_fit_stop = idx_start + n_fit
    t_test, u_test, y_test = wh2009_loader("train", scale=False, dataset_name="WienerHammerSysRamp.csv")

    # t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]
    N = t_test.shape[0]

    # Setup neural model structure
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_sizes=[hidden_size], hidden_acts=[nn.Tanh()]).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)

    model.load_state_dict(model_data["model"])
    model = model.to(dtype).to(device)

    n_param = sum(map(torch.numel, model.parameters()))
    n_fparam = sum(map(torch.numel, f_xu.parameters()))
    n_gparam = sum(map(torch.numel, g_x.parameters()))

    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u_test)
    x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_fparam, dtype=dtype)

    x_sim = []
    y_sim = []
    J_rows = []
    unc_var_step = []

    scaling_phi = np.sqrt(N)
    for time_idx in range(N):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        y_step = g_x(x_step)
        y_sim.append(y_step)

        # Jacobian of y wrt x
        basis_y = torch.eye(n_y).unbind()
        jacs_gx = [torch.autograd.grad(y_step, x_step, v, retain_graph=True)[0] for v in basis_y]
        J_gx = torch.stack(jacs_gx, dim=0)

        # Jacobian of y wrt theta
        jacs_gtheta = [torch.autograd.grad(y_step, g_x.parameters(), v, retain_graph=True) for v in basis_y]
        jacs_gtheta_f = [torch.cat([jac.ravel() for jac in jacs_gtheta[j]]) for j in range(n_y)]  # ravel jacobian rows
        J_gtheta = torch.stack(jacs_gtheta_f)  # stack jacobian rows to obtain a jacobian matrix

        # Eq. 14a in the fast adaptation paper (special case, f and g independently parameterized)
        phi_step_1 = J_gx @ s_step
        phi_step_2 = J_gtheta
        phi_step = torch.cat((phi_step_1, phi_step_2), axis=-1).t()
        unc_var_step.append(phi_step.t() @ P_post @ phi_step)  # output variance at time step

        # Current x
        # System update
        delta_x = 1.0 * f_xu(x_step, u_step)
        basis_x = torch.eye(n_x).unbind()

        # Jacobian of delta_x wrt x
        jacs_fx = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        J_fx = torch.stack(jacs_fx, dim=0)

        # Jacobian of delta_x wrt theta
        jacs_ftheta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in basis_x]
        jacs_ftheta_f = [torch.cat([jac.ravel() for jac in jacs_ftheta[j]]) for j in range(n_x)]  # ravel jacobian rows
        J_ftheta = torch.stack(jacs_ftheta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)

        s_step = s_step + J_fx @ s_step + J_ftheta  # Eq. 14a in the fast adaptation paper

    unc_var = torch.cat(unc_var_step)
    #J = torch.cat(J_rows).squeeze(-1)
    x_sim = torch.stack(x_sim)
    y_sim = torch.stack(y_sim)

    #%%
    y_sim = y_sim.detach().numpy()

    #%%
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t_test, y_test, 'k', label='$y$')
    ax[0].plot(t_test, y_sim, 'b', label='$\hat y$')
    ax[0].plot(t_test, y_test - y_sim, 'r', label='e')
    unc_std = np.sqrt(unc_var).reshape(-1, 1)
    ax[0].plot(t_test, 3 * (unc_std + sigma_noise), 'g', label='$3\sigma$')
    ax[0].plot(t_test, -3 * (unc_std + sigma_noise), 'g', label='$-3\sigma$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel(r"Time ($\mu_s$)")
    ax[0].set_ylabel("Current (A)")

    ax[1].plot(t_test, u, 'k', label='$v_{in}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel(r"Time ($\mu_s$)")
    ax[1].set_ylabel("Voltage (V)")

