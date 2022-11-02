import os
import numpy as np
import scipy
import torch
import matplotlib as mpl
mpl.use("TKAgg")
import torch.nn as nn
import torchid.ss.dt.models as models
from true_system import WHSys
from torchid.ss.dt.simulator import StateSpaceSimulator
from common_input_signals import multisine
import matplotlib.pyplot as plt
from torchid import metrics


# Truncated simulation error minimization method
if __name__ == '__main__':

    np.random.seed(42)

    font = {'size': 14,
            'family': 'serif'}

    mpl.rc('font', **font)

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
    n_skip = 64

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

    # %% Generate dataset
    fs = 51200
    ts = 1/fs

    sys = WHSys()

    SIGNAL = "MULTISINE_1"
    #SIGNAL = "CHIRP"


    ## Sine test ##
    if SIGNAL == "SINE":
        N = 5_000
        f = 3_000
        t_test = ts * np.arange(N).reshape(-1, 1)
        u_test = 1.0*np.sin(2*np.pi*f*t_test)

    ## multisine tests ##
    elif SIGNAL == "MULTISINE_1":  # RMSE = 5.6, FIT=98.1, surprise=0.34, coverage=99.2
        N = 5_000
        fmax = 2000  # equivalent to training data
        a = 0.4  # equivalent to training
        pmax = int(N*fmax/fs)
        u_test = a * multisine(N, 1, pmin=1, pmax=pmax, prule=lambda p: True)

    elif SIGNAL == "MULTISINE_2":  # RMSE = 5.9, FIT=97.7, surprise=0.43, coverage=98.6
        N = 5_000
        fmin = 1_000
        fmax = 2_000
        a = 0.4  # equivalent to training
        pmax = int(N*fmax/fs)
        pmin = int(N * fmin / fs)
        pmin = max(pmin, 1)
        u_test = a * multisine(N, 1, pmin=pmin, pmax=pmax, prule=lambda p: True)

    elif SIGNAL == "MULTISINE_3":  # RMSE = 32.5, FIT=93.9, surprise=2.10, coverage=96.1
        N = 5_000
        fmax = 2_000  # equivalent to training data
        a = 0.8  # double training
        pmax = int(N*fmax/fs)
        u_test = a * multisine(N, 1, pmin=1, pmax=pmax, prule=lambda p: True)

    elif SIGNAL == "MULTISINE_4":  # RMSE = 16.8, FIT=87.8, surprise=4.03, coverage=80.6
        N = 5_000
        fmin = 0
        fmax = 10_000
        a = 0.4  # equivalent to training
        pmax = int(N*fmax/fs)
        pmin = int(N * fmin / fs)
        pmin = max(pmin, 1)
        u_test = a * multisine(N, 1, pmin=pmin, pmax=pmax, prule=lambda p: True)

    elif SIGNAL == "RAMP":  # RMSE = 63.8, FIT=89.6, surprise=6.04
        N = 10_000
        fmax = 2000  # equivalent to training data
        a = 0.4  # equivalent to training
        pmax = int(N*fmax/fs)
        u_test = a * multisine(N, 1, pmin=1, pmax=pmax, prule=lambda p: True)
        u_test = u_test * np.linspace(0, 4, N)

    elif SIGNAL == "CHIRP":  # RMSE = 14.8, FIT=87.7, surprise=3.63
        N = 10_000
        f_max = 10_000
        f_min = 0
        t_test = ts * np.arange(N).reshape(-1, 1)
        u_test = 0.5*scipy.signal.chirp(t_test, f_min, t_test[-1], f_max, method='linear', phi=0, vertex_zero=True)

    # for all signals
    t_test = ts * np.arange(N).reshape(-1, 1)
    u_test = u_test.reshape(-1, 1)

    with torch.no_grad():
        u_torch = torch.tensor(u_test[None, ...], dtype=dtype)
        y_sim_torch = sys(u_torch)

    y_test_clean = y_sim_torch.numpy()[0, ...]
    y_test = y_test_clean + np.random.randn(*y_test_clean.shape) * sigma_noise

    # t_val, u_val, y_val = t_train[n_fit:] - t_train[n_fit], u_train[n_fit:], y_train[n_fit:]

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

    u = torch.tensor(u_test, dtype=dtype)
    x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_fparam, dtype=dtype)

    x_sim = []
    y_sim = []
    J_rows = []
    unc_var_step = []

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
    unc_var = unc_var.detach().numpy()
    unc_std = np.sqrt(unc_var).reshape(-1, 1)

    ppd_var = unc_var + var_noise
    ppd_std = np.sqrt(ppd_var)
    #%%
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 5.0))

    ax[0].plot(t_test, y_test, 'k', label='$\mathbf{y}^*$')
    ax[0].plot(t_test, y_sim, 'b', label='$\hat{\mathbf{y}}^*$')
    ax[0].fill_between(t_test.ravel(),
                     (y_sim + 3 * ppd_std).ravel(),
                     (y_sim - 3 * ppd_std).ravel(),
                     alpha=0.3,
                     color='c', label="99.7 % C.I.")
    ax[0].grid(True)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    factor = 1000
    ax[1].plot(t_test, factor*(y_test - y_sim), 'r', alpha=0.7, label='$\mathbf{e}$')
    ax[1].axhline(factor * 3 * sigma_noise, color="k", label="$+3 \sigma_e$")
    ax[1].axhline(factor * -3 * sigma_noise, color="k", label="$-3 \sigma_e$")
    ax[1].fill_between(t_test.ravel(),
                     factor * 3 * ppd_std.ravel(),
                     factor * -3 * ppd_std.ravel(),
                     alpha=0.3,
                     color='r', label="99.7 % C.I.")

    #ax[1].set_ylim(np.array([-0.01, 0.01])*factor)
    ax[1].grid()
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

    #ax[0].set_xlabel(r"Time ($s$)")
    #ax[0].set_ylabel("Voltage (V)")

    ax[2].plot(t_test, u, 'k', label='$\mathbf{u}$')
    ax[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), )
    # ax[2].legend(loc='upper right')
    ax[2].grid(True)
    ax[2].set_xlabel(r"Time ($s$)")

    ax[0].set_ylabel("Voltage (V)")
    ax[1].set_ylabel("Voltage (mV)")
    ax[2].set_ylabel("Voltage (V)")

    if SIGNAL in ["MULTISINE_1", "MULTISINE_2", "MULTISINE_3", "MULTISINE_4"]:
        ax[2].set_xlim([0.04, 0.06])  # for MS_2
        ax[2].set_xticks([0.04, 0.045, 0.05, 0.055, 0.06])
    else:
        ax[2].set_xlim([t_test[n_skip], t_test[-1]])

    if SIGNAL == "MULTISINE_3":
        ax[1].set_ylim([-650, 650])  # for MS_2
        ax[1].set_yticks([-600, -400, -200, 0,
                          200,  400,  600])
    else:
        ax[1].set_ylim([-70, 70])  # for MS_2
    #ax[2].set_xlim([t_test[1000], t_test[1500]])

    plt.tight_layout()
    plt.savefig(f"{SIGNAL}.pdf")
    #ax[3].plot(t_test, unc_std / (np.abs(y_sim) + 1e-2))
    #%%

    y_metrics = y_test[n_skip:]
    y_sim_metrics = y_sim[n_skip:]

    e_rms = factor * metrics.rmse(y_metrics, y_sim_metrics)[0]
    fit_idx = metrics.fit_index(y_metrics, y_sim_metrics)[0]
    r_sq = metrics.r_squared(y_metrics, y_sim_metrics)[0]

    print(f"Signal: {SIGNAL}")
    print(f"RMSE: {e_rms:.1f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")

    #%%
    def rms(sig, time_axis=0):
        return np.sqrt(np.mean(sig ** 2, axis=time_axis))

    def mabs(sig, time_axis=0):
        return np.mean(np.abs(sig), axis=time_axis)

    #surprise = (100*rms(unc_std) / rms(y_sim))[0]
    surprise = (100*mabs(unc_std) / mabs(y_sim))[0]
    print(f"surprise: {surprise:.5f}")

    err = y_test - y_sim
    in_band = np.abs(err) < 3 * ppd_std
    coverage = np.sum(in_band)/N * 100
    print(f"coverage: {coverage:.1f}%")
