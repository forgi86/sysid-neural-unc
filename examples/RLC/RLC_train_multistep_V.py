import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchid.ss.dt.models as models
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator, FeedForwardStateEstimator
from torchid.datasets import SubsequenceDataset
from loader import rlc_loader


# Truncated simulation error minimization method
if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    no_cuda = False  # no GPU, CPU only training
    threads = 6  # max number of CPU threads

    # Overall parameters
    epochs = 500  # training epochs
    seq_sim_len = 128  # simulation sequence length
    seq_est_len = 32  # estimation sequence length
    batch_size = 256  # batch size q
    lr = 1e-4  # learning rate
    n_fit = 5000
    hidden_sizes = [32, 16]
    hidden_acts = [nn.Tanh(), nn.Tanh()]
    n_x = 2

    # CPU/GPU resources
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(threads)

    # Load dataset
    t, u, y, _ = rlc_loader("train", "nl", noise_std=0.05, n_data=n_fit, output='V_C')  # state not used

    # Setup neural model structure
    f_xu = models.NeuralLinStateUpdate(n_x=2, n_u=1,
                                       hidden_sizes=hidden_sizes, hidden_acts=hidden_acts).to(device)
    g_x = models.ChannelsOutput(channels=[0]).to(device)  # output is channel 0
    model = StateSpaceSimulator(f_xu, g_x).to(device)
    estimator = FeedForwardStateEstimator(n_u=1, n_y=1, n_x=2, seq_len=32, hidden_size=64).to(device)

    load_len = seq_sim_len + seq_est_len
    train_dataset = SubsequenceDataset(torch.from_numpy(u), torch.from_numpy(y), subseq_len=load_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': estimator.parameters(), 'lr': lr},
    ], lr=lr)

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    start_time = time.time()

    model.f_xu.disable_nl()
    model.f_xu.freeze_nl()
    # Training loop
    for epoch in range(epochs):

        if epoch == 100:
            model.f_xu.enable_nl()
            model.f_xu.unfreeze_nl()
            model.f_xu.freeze_lin()

        for batch_idx, (batch_u, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_u = batch_u.transpose(0, 1).to(device)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1).to(device)  # transpose to time_first

            # Estimate initial state
            batch_u_est = batch_u[:seq_est_len]
            batch_y_est = batch_y[:seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            # Simulate
            batch_u_fit = batch_u[seq_est_len:]
            batch_y_fit = batch_y[seq_est_len:]
            batch_y_sim = model(batch_x0, batch_u_fit)

            # Compute loss
            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)

            # Statistics
            LOSS.append(loss.item())

            # Optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} | Train Loss {loss:.4f} ')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    #%%

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model = model.to("cpu")
    estimator = estimator.to("cpu")
    model_filename = "ss_model_ms_V.pt"
    torch.save({
                "n_x": n_x,
                "n_feat": hidden_sizes,
                "model": model.state_dict(),
                "estimator": estimator.state_dict()
                },
               os.path.join("models", model_filename))

    #%% Simulate
    with torch.no_grad():
        u_v = torch.tensor(u[:, None, :])
        y_v = torch.tensor(y[:, None, :])
        x0 = torch.zeros(1, n_x, dtype=torch.float32)  # initial state set to 0 for simplicity
        y_sim = model(x0, u_v)

    #%% Test
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(y_v[:, 0, 0], 'k', label='meas')
    ax.grid(True)
    ax.plot(y_sim[:, 0, 0], 'b', label='sim')
    #ax.plot(y_v[:, 0, 0] - y_sim[:, 0, 0], 'r', label='err')

