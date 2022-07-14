import time
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("TKAgg")
import torch.nn as nn
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from loader import wh2009_loader
from diffutils import parameter_jacobian


if __name__ == '__main__':

    model_filename = "model.pt"
    model_data = torch.load(os.path.join("models", model_filename))

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
    n_fit = 2000

    no_cuda = True  # no GPU, CPU only training
    dtype = torch.float64
    threads = 6  # max number of CPU threads

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

    # Compute Jacobian
    time_start = time.time()
    x_0 = torch.zeros((1, n_x), dtype=dtype)
    u_v = torch.tensor(u_fit)[:, None, :]
    J = parameter_jacobian(model, x_0, u_v, vectorize=True, flatten=True)
    time_jac = time.time() - time_start
    print(f"\nSlow jacobian computation time: {time_jac:.2f}")
    torch.save(J, "J_slow.pt")
