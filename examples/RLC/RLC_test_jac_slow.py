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
import functorch
from diffutils import parameter_jacobian


class StateSpaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(StateSpaceWrapper, self).__init__()
        self.model = model

    def forward(self, u_in):
        x_0 = torch.zeros((1, n_x), dtype=torch.float32)
        x_sim_torch = self.model(x_0, u_in)
        y_out = x_sim_torch[:, [0]]
        return y_out


if __name__ == '__main__':

    model_filename = "ss_model_ms.pt"
    model_data = torch.load(os.path.join("models", model_filename))
    hidden_sizes = model_data["hidden_sizes"]
    hidden_acts = model_data["hidden_acts"]

    # Column names in the dataset
    t, u, y, x = rlc_loader("test", "nl", noise_std=0.0, output='I_L')
    n_x = x.shape[-1]
    ts = t[1, 0] - t[0, 0]

    # Setup neural model structure and load fitted model parameters
    f_xu = models.NeuralLinStateUpdate(n_x=2, n_u=1,
                                       hidden_sizes=hidden_sizes, hidden_acts=hidden_acts)
    g_x = models.ChannelsOutput(channels=[0])  # output mapping corresponding to channel 0
    model = StateSpaceSimulator(f_xu, g_x)
    model.load_state_dict(model_data["model"])

    # Evaluate the model in open-loop simulation against validation data
    x_0 = torch.zeros((1, n_x), dtype=torch.float32)
    u_v = torch.tensor(u)[:, None, :]
    with torch.no_grad():
        y_sim = model(x_0, u_v).squeeze(1)
    y_sim = y_sim.detach().numpy()

    J = parameter_jacobian(model, x_0, u_v, vectorize=True, flatten=True)
    # Plot results
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 5.5))

    ax[0].plot(t, y, 'k',  label='$v_C$')
    ax[0].plot(t, y_sim, 'b',  label='$\hat v_C$')
    ax[0].plot(t, y-y_sim, 'r',  label='e')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time (mu_s)")
    ax[0].set_ylabel("Voltage (V)")

    ax[1].plot(t, u, 'k',  label='$v_{in}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time (mu_s)")
    ax[1].set_ylabel("Voltage (V)")

    plt.show()

    # R-squared metrics
    R_sq = metrics.r_squared(y, y_sim)
    print(f"R-squared: {R_sq}")

    fit = metrics.fit_index(y, y_sim)
    print(f"fit index: {fit}")

