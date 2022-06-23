import os
import numpy as np
import torch
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid import metrics
from loader import rlc_loader
from diffutils import parameter_jacobian


if __name__ == '__main__':

    model_filename = "ss_model_ms.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    hidden_sizes = model_data["hidden_sizes"]
    hidden_acts = model_data["hidden_acts"]

    # Column names in the dataset
    t, u, y, x = rlc_loader("test", "nl", noise_std=0.0, output='I_L')
    n_x = x.shape[-1]
    ts = t[1, 0] - t[0, 0]
    seq_len = t.shape[0]
    noise_std = 0.05

    # Setup neural model structure and load fitted model parameters
    f_xu = models.NeuralLinStateUpdate(n_x=2, n_u=1,
                                       hidden_sizes=hidden_sizes, hidden_acts=hidden_acts)
    g_x = models.ChannelsOutput(channels=[0])  # output mapping corresponding to channel 0
    model = StateSpaceSimulator(f_xu, g_x)
    model.load_state_dict(model_data["model"])

    n_param = sum(map(torch.numel, f_xu.parameters()))
    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u)
    x_step = torch.zeros((n_x), dtype=torch.float32, requires_grad=True)
    s_step = torch.zeros(n_x, n_param)

    x_sim = []
    J_rows = []

    for time_idx in range(seq_len):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        phi_step = s_step[[0], :]  # Special case of (14b), output = first state
        J_rows.append(phi_step)

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
    #x_0 = torch.zeros((1, n_x), dtype=torch.float32)
    #u_v = torch.tensor(u)[:, None, :]
    #J_slow = parameter_jacobian(model, x_0, u_v, vectorize=True, flatten=True)
    #%%

    I_theta = J.transpose() @ J/noise_std
    P_theta = np.linalg.inv(I_theta)
    P_y = J @ P_theta @ J.transpose()

    W, V = np.linalg.eig(I_theta)
    #plt.plot(W.real, W.imag, "*")

    #%%
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t, y, 'k',  label='$v_C$')
    ax[0].plot(t, y_sim, 'b',  label='$\hat v_C$')
    ax[0].plot(t, y-y_sim, 'r',  label='e')
    ax[0].plot(t, 6*np.sqrt(np.diag(P_y)), 'g',  label='$\sigma$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time (mu_s)")
    ax[0].set_ylabel("Current (A)")

    ax[1].plot(t, u, 'k',  label='$v_{in}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time (mu_s)")
    ax[1].set_ylabel("Voltage (V)")

    plt.show()

