import os
import matplotlib
matplotlib.use("TKAgg")
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import WHNet3
from torchid import metrics
import control

if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    model_name = "model_WH3"

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))
    signal_num = 0  # signal used for test (nominal model trained on signal 0)

    # Extract data
    y_meas = np.array(df_X[[f"y{signal_num}"]], dtype=np.float32)
    u = np.array(df_X[["u"]], dtype=np.float32)
    fs = 51200
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    test_start = 0
    N_test = 10000
    y_meas = y_meas[test_start:test_start+N_test, [0]]
    u = u[test_start:test_start+N_test, [0]]
    t = t[test_start:test_start+N_test]

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)


    # In[Instantiate models]

    # Create models
    model = WHNet3()
    model_folder = os.path.join("models", model_name)
    model.load_state_dict(torch.load(os.path.join(model_folder, "model_wh3.pt")))

    # In[Simulate]
    with torch.no_grad():
        y_sim_torch = model(u_torch)

    y_sim = y_sim_torch.numpy()[0, ...]

    # In[Metrics]
    R_sq = metrics.r_squared(y_meas, y_sim)[0]
    rmse = metrics.error_rmse(y_meas, y_sim)[0]

    print(f"R-squared metrics: {R_sq}")
    print(f"RMSE metrics: {rmse}")

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_sim, 'b', label="$\hat y$")
    plt.legend()

    # In[Analysis]
    n_imp = 128
    G1 = model.G1
    G1_num, G1_den = G1.get_tfdata()
    G1_sys = control.TransferFunction(G1_num, G1_den, ts)
    plt.figure()
    plt.title("$G_1$ impulse response")
    _, y_imp = control.impulse_response(G1_sys, np.arange(n_imp) * ts)
    #    plt.plot(G1_num)
    plt.plot(y_imp)
    plt.savefig(os.path.join("models", model_name, "G1_imp.pdf"))
    plt.figure()
    mag_G1, phase_G1, omega_G1 = control.bode(G1_sys, omega_limits=[1e2, 1e5])
    plt.suptitle("$G_1$ bode plot")
    plt.savefig(os.path.join("models", model_name, "G1_bode.pdf"))

    G2 = model.G2
    G2_num, G2_den = G2.get_tfdata()
    G2_sys = control.TransferFunction(G2_num, G2_den, ts)
    plt.figure()
    plt.title("$G_2$ impulse response")
    _, y_imp = control.impulse_response(G2_sys, np.arange(n_imp) * ts)
    plt.plot(y_imp)
    plt.savefig(os.path.join("models", model_name, "G1_imp.pdf"))
    plt.figure()
    mag_G2, phase_G2, omega_G2 = control.bode(G2_sys, omega_limits=[1e2, 1e5])
    plt.suptitle("$G_2$ bode plot")
    plt.savefig(os.path.join("models", model_name, "G2_bode.pdf"))

    F_nl = model.F_nl
    y1_lin_min = -5.0  #np.min(y1_lin)
    y1_lin_max = 5.0  # np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))
        y_lin = model.G1(u_torch)
        y_nl = F_nl(y_lin)

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(y_lin.squeeze(), y_nl.squeeze(), '*')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity output (-)')
    plt.grid(True)
