import os
import torch
import pandas as pd
import numpy as np
from models import WHNet3
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import time
import torchid.metrics


# In[Main]
if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Settings]
    lr_ADAM = 2e-4
    lr_BFGS = 1e0
    num_iter_ADAM = 100000  # ADAM iterations 20000
    num_iter_BFGS = 10  # final BFGS iterations
    msg_freq = 100
    n_skip = 5000
    n_fit = 20000
    decimate = 1
    n_batch = 1
    n_b = 8
    n_a = 8
    model_name = "model_WH3"

    num_iter = num_iter_ADAM + num_iter_BFGS

    # In[Column names in the dataset]
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # In[Load dataset]
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y = np.array(df_X[COL_Y], dtype=np.float32)  # batch, time, channel
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype=np.float32)
    N = y.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Fit data]
    y_fit = y[0:n_fit:decimate]
    u_fit = u[0:n_fit:decimate]
    t_fit = t[0:n_fit:decimate]

    # In[Prepare training tensors]
    u_fit_torch = torch.tensor(u_fit[None, :, :], dtype=torch.float, requires_grad=False)
    y_fit_torch = torch.tensor(y_fit[None, :, :], dtype=torch.float)

    # In[Prepare model]
    model = WHNet3()

    # In[Setup optimizer]
    optimizer_ADAM = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_ADAM},
    ], lr=lr_ADAM)

    optimizer_LBFGS = torch.optim.LBFGS(list(model.parameters()), lr=lr_BFGS)


    def closure():
        optimizer_LBFGS.zero_grad()

        # Simulate
        y_hat = model(u_fit_torch)

        # Compute fit loss
        err_fit = y_fit_torch[:, n_skip:, :] - y_hat[:, n_skip:, :]
        loss = torch.mean(err_fit**2)*1000

        # Backward pas
        loss.backward()
        return loss


    # In[Train]
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        if itr < num_iter_ADAM:
            msg_freq = 10
            loss_train = optimizer_ADAM.step(closure)
        else:
            msg_freq = 10
            loss_train = optimizer_LBFGS.step(closure)

        LOSS.append(loss_train.item())
        if itr % msg_freq == 0:
            with torch.no_grad():
                RMSE = torch.sqrt(loss_train)
            print(f'Iter {itr} | Fit Loss {loss_train:.6f} | RMSE:{RMSE:.4f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Save model]
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), os.path.join(model_folder, "model_wh3.pt"))


    # In[Simulate one more time]
    with torch.no_grad():
        y_hat = model(u_fit_torch)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]

    # In[Plot]
    plt.figure()
    plt.plot(t_fit, y_fit, 'k', label="$y$")
    plt.plot(t_fit, y_hat, 'b', label="$\hat y$")
    plt.legend()

    # In[Plot loss]
    plt.figure()
    plt.plot(LOSS)
    plt.grid(True)

    # In[Plot]
    R_sq = torchid.metrics.r_squared(y_hat, y_fit)[0]
    e_rms = torchid.metrics.rmse(y_hat, y_fit)[0]

    print(f"R-squared metrics: {R_sq}")
    print(f"RMSE metrics: {e_rms}")







