import os
import torch
import torchid.ss.dt.models as models
import torchid.ss.dt.estimators as estimators
from torchid.ss.dt.simulator import StateSpaceSimulator
from torch.utils.data import DataLoader
from torchid.datasets import SubsequenceDataset
from loader import wh2009_loader, wh2009_scaling
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from torchid import metrics


if __name__ == '__main__':

    # model_data = torch.load(os.path.join("models", "model.pt"))
    #model_data = torch.load(os.path.join("models", "doe1", "model_1.pt"))
    #model_data = torch.load(os.path.join("models", "doe2", "model_1.pt"), map_location=torch.device('cpu'))
    #model_data = torch.load(os.path.join("models", "doe2", "model_123.pt"), map_location=torch.device('cpu'))  # best
    #model_data = torch.load(os.path.join("models", "doe2", "model_276.pt"), map_location=torch.device('cpu'))  # worst
    #model_data = torch.load(os.path.join("models", "doe2", "model_169.pt"), map_location=torch.device('cpu'))  # short lstm
    #model_data = torch.load(os.path.join("models", "doe3", "model_279.pt"), map_location=torch.device('cpu'))

    #model_data = torch.load(os.path.join("models", "doe5", "model_420.pt"), map_location=torch.device('cpu'))  # worst
    #model_data = torch.load(os.path.join("models", "doe5", "model_113.pt"), map_location=torch.device('cpu'))  # worst

    model_data = torch.load(os.path.join("models", "doe5", "model_5.pt"), map_location=torch.device('cpu'))  # worst
    #model_data = torch.load(os.path.join("models", "doe5", "model_578.pt"), map_location=torch.device('cpu'))  # worst


    n_x = model_data["n_x"]
    n_y = model_data["n_y"]
    n_u = model_data["n_u"]
    args = model_data["args"]

    torch.manual_seed(args.seed)

    # Derived parameters
    if "est_frac" in args and args.est_frac is not None:
        seq_est_len = int(args.seq_fit_len * args.est_frac)

    if "seq_est_len" in args and args.seq_est_len is not None:
        seq_est_len = args.seq_est_len
    backward_est = True if args.est_direction == "backward" else False
    if backward_est:
        load_len = max(args.seq_fit_len, seq_est_len)
    else:
        load_len = args.seq_fit_len + seq_est_len

    # Load dataset
    t, u, y = wh2009_loader("train", scale=True)
    y_mean, y_std = wh2009_scaling()

    dataset = SubsequenceDataset(u, y, subseq_len=load_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    u_t = torch.tensor(u[:, None, :])
    y_t = torch.tensor(y[:, None, :])

    #%% Load models and parameters
    f_xu = models.NeuralLinStateUpdate(n_x, n_u, hidden_size=args.hidden_size)
    g_x = models.NeuralLinOutput(n_x, n_u, hidden_size=args.hidden_size)  #LinearOutput(n_x, n_y)
    model = StateSpaceSimulator(f_xu, g_x)
    if args.est_type == "LSTM":
        estimator = estimators.LSTMStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                  hidden_size=args.est_hidden_size,
                                                  flipped=backward_est)
    elif args.est_type == "FF":
        estimator = estimators.FeedForwardStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x,
                                                         hidden_size=args.est_hidden_size,
                                                         seq_len=seq_est_len)
    elif args.est_type == "ZERO" or args.est_type == "RAND":
        estimator = estimators.ZeroStateEstimator(n_u=n_y, n_y=n_y, n_x=n_x)
    else:
        raise ValueError("Wrong estimator type. Possible values: LSTM|FF|ZERO")

    model.load_state_dict(model_data["model"])
    estimator.load_state_dict(model_data["estimator"])

    val_loss = 0.0
    with torch.no_grad():
        for batch_u, batch_y in loader:
            batch_u = batch_u.transpose(0, 1)  # transpose to time_first
            batch_y = batch_y.transpose(0, 1)  # transpose to time_first

            batch_est_u = batch_u[:seq_est_len]
            batch_est_y = batch_y[:seq_est_len]
            batch_x0 = estimator(batch_est_u, batch_est_y)

            batch_u_est = batch_u[:args.seq_est_len]
            batch_y_est = batch_y[:args.seq_est_len]
            batch_x0 = estimator(batch_u_est, batch_y_est)

            batch_y_fit = batch_y[args.seq_est_len:]

            if args.est_type not in ["ZERO", "RAND"]:  # for not-dummy estimators
                batch_u_fit = batch_u[args.seq_est_len:]
            else:
                batch_u_fit = batch_u

            batch_y_sim = model(batch_x0, batch_u_fit)

            batch_y_sim_full = batch_y_sim.clone()
            if args.est_type not in ["ZERO", "RAND"]:
                batch_y_sim_full = torch.cat((torch.nan*torch.zeros((args.seq_est_len, args.batch_size, 1)), batch_y_sim_full), 0)

            # Compute fit loss
            # Compute fit loss
            if args.est_type in ["ZERO", "RAND"]:  # for dummy estimators
                batch_y_sim = batch_y_sim[args.seq_est_len:]

            loss = torch.nn.functional.mse_loss(batch_y_fit, batch_y_sim)
            val_loss += loss.item()

    val_loss = val_loss/len(loader)

    print(f"Val loss: {val_loss:.3f}")
    #%%
    batch_y_sim_full_np = batch_y_sim_full.squeeze(-1).transpose(0, 1).numpy()
    batch_y_np = batch_y.squeeze(-1).transpose(0, 1).numpy()

    examples = 3
    fig, ax = plt.subplots(examples, 1, sharex=True)
    for idx in range(examples):
        ax[idx].plot(batch_y_np[idx], 'k')
        ax[idx].plot(batch_y_sim_full_np[idx], 'b--')

        #ax[idx].plot(batch_y_np[idx] - batch_y_sim_full_np[idx], 'r-.')
        if idx == examples - 1:
            ax[idx].set_xlabel("estimation/fitting sequence index (-)")
        #ax[idx].grid(axis="y")
    plt.savefig("subseq_best.pdf")
