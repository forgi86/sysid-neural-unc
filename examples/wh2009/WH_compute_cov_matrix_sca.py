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
    idx_start = 5000
    n_fit = 10000

    no_cuda = True  # no GPU, CPU only training
    dtype = torch.float64
    threads = 6  # max number of CPU threads
    beta_prior = 0.01  # precision (1/var) of the prior on theta
    sigma_noise = 0.02  # noise variance (could be learnt instead)

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
    t_train, u_train, y_train = wh2009_loader("train", scale=True)
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

    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u_fit)
    x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_fparam, dtype=dtype)

    scaling_H = 1/(N * beta_noise)
    scaling_P = 1/scaling_H
    scaling_phi = np.sqrt(beta_noise * scaling_H)  # np.sqrt(N)

    # negative Hessian of the log-prior
    H_prior = torch.eye(n_param, dtype=dtype) * beta_prior * scaling_H
    P_prior = torch.eye(n_param, dtype=dtype) / beta_prior * scaling_P
    P_step = P_prior  # prior parameter covariance
    H_step = torch.zeros((n_param, n_param), dtype=dtype)
    #H_step = torch.eye(n_param) * beta_prior/scaling_H  # prior inverse parameter covariance

    x_sim = []
    y_sim = []
    J_rows = []

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

        # Eq. 14a in the paper (special case, f and g independently parameterized)
        phi_step_1 = J_gx @ s_step
        phi_step_2 = J_gtheta
        phi_step = torch.cat((phi_step_1, phi_step_2), axis=-1).t() * scaling_phi

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
        jacs_fx = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        J_fx = torch.stack(jacs_fx, dim=0)

        # Jacobian of delta_x wrt theta
        jacs_ftheta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in basis_x]
        jacs_ftheta_f = [torch.cat([jac.ravel() for jac in jacs_ftheta[j]]) for j in range(n_x)]  # ravel jacobian rows
        J_ftheta = torch.stack(jacs_ftheta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)

        s_step = s_step + J_fx @ s_step + J_ftheta  # Eq. 14a in the paper

    J = torch.cat(J_rows).squeeze(-1)
    x_sim = torch.stack(x_sim)
    y_sim = torch.stack(y_sim)

    # information matrix (or approximate negative Hessian of the log-likelihood)
    H_post = H_prior + H_step
    P_post = torch.linalg.pinv(H_post)
    #P_step = P_step.numpy()
    #H_step = H_step.numpy()

    torch.save({
        "H_post": H_post,
        "P_post": P_post,
        "scaling_P": scaling_P,
        "scaling_H": scaling_H,
        "scaling_phi": scaling_phi
    }, os.path.join("models", "covariance.pt"))

    #%%
    P_y = J @ (P_post/scaling_P) @ J.t()/(scaling_phi**2)
    W, V = np.linalg.eig(H_post)
    #plt.plot(W.real, W.imag, "*")

    #%%
    y_sim = y_sim.detach().numpy()

    #%%
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t_fit, y_fit, 'k',  label='$y$')
    ax[0].plot(t_fit, y_sim, 'b',  label='$\hat y$')
    ax[0].plot(t_fit, y_fit-y_sim, 'r',  label='e')
    unc_std = np.sqrt(np.diag(P_y)).reshape(-1, 1)
    ax[0].plot(t_fit, 6*unc_std, 'g',  label='$6\sigma$')
    ax[0].plot(t_fit, -6*unc_std, 'g',  label='$-6\sigma$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel(r"Time ($\mu_s$)")
    ax[0].set_ylabel("Current (A)")

    ax[1].plot(t_fit, u, 'k',  label='$v_{in}$')
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

    U, S, VH = np.linalg.svd(H_post, hermitian=True)
    V = VH.transpose()
    # Vk = V[:10, :] an identifiable linear combination of parameters

    E, Q = np.linalg.eig(H_post)  # equivalent (up to sign permulations)
    E = np.real(E)
    Q = np.real(Q)
    plt.figure()
    plt.plot(E, "*")

    #%%
    plt.figure()
    plt.suptitle("Covariance Inverse")
    plt.imshow(H_step)
    plt.colorbar()
    plt.show()

    #%%

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

