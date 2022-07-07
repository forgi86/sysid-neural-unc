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

    model_filename = "ss_model_ms_V.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    hidden_sizes = model_data["hidden_sizes"]
    hidden_acts = model_data["hidden_acts"]

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Constants
    n_x = 6
    n_u = 1
    n_y = 1
    est_hidden_size = 16
    hidden_size = 16
    n_fit = 10000

    no_cuda = True  # no GPU, CPU only training
    dtype = torch.float64
    threads = 6  # max number of CPU threads
    n_fit = -1  # all points
    beta_prior = 0.01  # precision (1/var) of the prior on theta
    sigma_noise = 0.05  # noise variance (could be learnt instead)

    var_noise = sigma_noise**2
    beta_noise = 1/var_noise
    var_prior = 1/beta_prior

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    # %% Load dataset
    t_train, u_train, y_train = wh2009_loader("train", scale=True)
    t_fit, u_fit, y_fit = t_train[:n_fit], u_train[:n_fit], y_train[:n_fit]
    # t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

    # Setup neural model structure
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_sizes=[hidden_size], hidden_acts=[nn.Tanh()]).to(device)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=hidden_size).to(device)
    model = StateSpaceSimulator(f_xu, g_x).to(device)

    model.load_state_dict(model_data["model"])
    model = model.to(dtype).to(device)

    n_param = sum(map(torch.numel, f_xu.parameters()))
    # Evaluate the model in open-loop simulation against validation data

    #seq_len = 10000

    u = torch.from_numpy(u)
    x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_param, dtype=dtype)

    scaling_H = 1/(seq_len * beta_noise)
    scaling_P = 1/scaling_H
    scaling_phi = np.sqrt(beta_noise * scaling_H)

    # negative Hessian of the log-prior
    H_prior = torch.eye(n_param, dtype=dtype) * beta_prior * scaling_H
    P_step = torch.eye(n_param, dtype=dtype) / beta_prior/scaling_H  # prior parameter covariance
    H_step = torch.zeros((n_param, n_param), dtype=dtype)
    #H_step = torch.eye(n_param) * beta_prior/scaling_H  # prior inverse parameter covariance

    x_sim = []
    J_rows = []

    for time_idx in range(seq_len):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        phi_step = s_step[[0], :].t() * scaling_phi  # Special case of (14b), output = first state

        J_rows.append(phi_step.t())
        H_step = H_step + phi_step @ phi_step.t()

        den = 1 + phi_step.t() @ P_step @ phi_step
        P_tmp = - (P_step @ phi_step @ phi_step.t() @ P_step)/den
        P_step = P_step + P_tmp

        # Current x
        # System update
        delta_x = 1.0 * f_xu(x_step, u_step)
        basis_x = torch.eye(n_x).unbind()

        # Jacobian of delta_x wrt x
        jacs_x = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        J_x = torch.stack(jacs_x, dim=0)

        # Jacobian of delta_x wrt theta
        jacs_theta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in basis_x]
        jacs_theta_f = [torch.cat([jac.ravel() for jac in jacs_theta[j]]) for j in range(n_x)]  # ravel jacobian rows
        J_theta = torch.stack(jacs_theta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)

        s_step = s_step + J_x @ s_step + J_theta  # Eq. 14a in the paper

    J = torch.cat(J_rows).squeeze(-1)
    x_sim = torch.stack(x_sim)
    y_sim = x_sim[:, [0]].detach().numpy()

    # information matrix (or approximate negative Hessian of the log-likelihood)
    H_lik = J.t() @ J
    H_post = H_prior + H_step

    #P_step = P_step.numpy()
    #H_step = H_step.numpy()


    #%%
    P_post = torch.linalg.inv(H_post) * scaling_P
    P_y = J @ P_step @ J.t()
    W, V = np.linalg.eig(H_lik)
    #plt.plot(W.real, W.imag, "*")

    #%%
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t, y, 'k',  label='$v_C$')
    ax[0].plot(t, y_sim, 'b',  label='$\hat v_C$')
    ax[0].plot(t, y-y_sim, 'r',  label='e')
    ax[0].plot(t, 6*np.sqrt(np.diag(P_y)), 'g',  label='$3\sigma$')
    ax[0].plot(t, -6*np.sqrt(np.diag(P_y)), 'g',  label='$-3\sigma$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel(r"Time ($\mu_s$)")
    ax[0].set_ylabel("Current (A)")

    ax[1].plot(t, u, 'k',  label='$v_{in}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel(r"Time ($\mu_s$)")
    ax[1].set_ylabel("Voltage (V)")


    #%%
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Covariance")
    ax[0].matshow(P_post)
    ax[1].set_title("Covariance Inverse")
    ax[1].matshow(H_post)

    U, S, V = np.linalg.svd(H_post)
    plt.figure()
    plt.plot(S, "*")

    plt.figure()
    plt.suptitle("Covariance Inverse")
    plt.imshow(H_lik)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance Inverse Recursive")
    plt.imshow(H_step)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance Inverse Error")
    plt.imshow(H_lik - H_step)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance")
    plt.imshow(P_post)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance Recursive")
    plt.imshow(P_step)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance error")
    plt.imshow(P_post - P_step)
    plt.colorbar()
    plt.show()

