import os
import numpy as np
import torch
import matplotlib
matplotlib.use("TKAgg")
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import rlc_loader
import matplotlib.pyplot as plt


# Truncated simulation error minimization method
if __name__ == '__main__':

    model_filename = "ss_model_ms.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    hidden_sizes = model_data["hidden_sizes"]
    hidden_acts = model_data["hidden_acts"]

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    no_cuda = True  # no GPU, CPU only training
    threads = 6  # max number of CPU threads
    n_fit = -1  # all points
    tau_prior = 0.1  # precision (1/var) of the prior on theta
    sigma_noise = 0.05  # noise variance (could be learnt instead)

    var_noise = sigma_noise**2
    beta_noise = 1/var_noise

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    t, u, y, x = rlc_loader("train", "nl", noise_std=sigma_noise, n_data=n_fit, output='I_L')  # state not used

    n_x = x.shape[-1]
    ts = t[1, 0] - t[0, 0]
    seq_len = t.shape[0]

    # Setup neural model structure
    f_xu = models.NeuralLinStateUpdate(n_x=2, n_u=1,
                                       hidden_sizes=hidden_sizes, hidden_acts=hidden_acts).to(device)
    g_x = models.ChannelsOutput(channels=[0]).to(device)  # output is channel 0
    model = StateSpaceSimulator(f_xu, g_x).to(device)

    model.load_state_dict(model_data["model"])

    n_param = sum(map(torch.numel, f_xu.parameters()))
    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u)
    x_step = torch.zeros(n_x, dtype=torch.float32, requires_grad=True)
    s_step = torch.zeros(n_x, n_param)
    P_step = torch.eye(n_param) * 1 / tau_prior  # prior parameter covariance
    H_step = torch.zeros((n_param, n_param), dtype=torch.float32) #torch.eye(n_param) * tau_prior  # prior inverse parameter covariance

    x_sim = []
    J_rows = []

    for time_idx in range(3):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        phi_step = s_step[[0], :]  # Special case of (14b), output = first state

        J_rows.append(phi_step)
        H_step = H_step + phi_step.t() @ phi_step * beta_noise

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


    J = torch.cat(J_rows).squeeze(-1).numpy()
    x_sim = torch.stack(x_sim)
    y_sim = x_sim[:, [0]].detach().numpy()

    #%%

    # negative Hessian of the log-prior
    H_prior = np.eye(n_param) * tau_prior
    # information matrix (or approximate negative Hessian of the log-likelihood)
    H_lik = J.transpose() @ J * beta_noise
    H_post = H_prior + H_lik
    P_post = np.linalg.inv(H_lik)
    P_y = J @ P_post @ J.transpose()

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

    plt.show()

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
    plt.imshow(H_post)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.suptitle("Covariance Inverse")
    plt.imshow(H_post - H_step.numpy())
    plt.colorbar()
    plt.show()

