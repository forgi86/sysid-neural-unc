import os
import matplotlib
matplotlib.use("TKAgg")
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import WHSys
import control


if __name__ == '__main__':

    torch.manual_seed(42)
    np.random.seed(42)

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    model_name = "model_WH3"

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))
    signal_num = 0  # signal used for test (nominal model trained on signal 0)

    # Extract data
    y_meas = np.array(df_X[["yBenchMark"]], dtype=np.float32)  # batch, time, channel
    u = np.array(df_X[["uBenchMark"]], dtype=np.float32)

    #u = u[20_000:40_000, :]
    #y_meas = y_meas[20_000:40_000, :]
    #u_scale = (2 * np.arange(len(u)) / len(u)).reshape(-1, 1)
    u = u[::10]  # * u_scale
    y_meas = y_meas[::10]
    fs = 51200
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts
    sigma_noise = 5e-3

    # Prepare data
    u_torch = torch.tensor(u[None, ...], dtype=torch.float, requires_grad=False)

    # In[Instantiate models]
    model = WHSys()

    # In[Simulate]
    with torch.no_grad():
        y_sim_torch = model(u_torch)

    y_sim = y_sim_torch.numpy()[0, ...]
    y_sim_noise = y_sim + np.random.randn(*y_sim.shape) * sigma_noise

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_sim, 'b', label="$\hat y$")
    plt.plot(t, y_sim_noise, 'r', label="$\hat y$ noise")
    plt.legend()

    plt.figure()
    plt.plot(u)


    #%%

    df_X = pd.DataFrame({"uBenchMark": u.ravel(),
                         "yBenchMark": y_sim_noise.ravel(),
                         "yBenchMarkClean": y_sim.ravel(), "fs": fs})
    df_X = df_X[["uBenchMark", "yBenchMark", "yBenchMarkClean", "fs"]]
    df_X.to_csv(os.path.join("data", "WienerHammerSysDyn.csv"), index=False)



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
    mag_G1, phase_G1, omega_G1 = control.bode(G1_sys, omega_limits=[1e3, 1e5])
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

    #%%
    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))
        y_lin = model.G1(u_torch)
        y_nl = F_nl(y_lin)
        out_nl_approx = -torch.nn.functional.elu(-torch.from_numpy(10 / 11 * (in_nl - 0.0)), alpha=1.0) + 0.0

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(y_lin.squeeze(), y_nl.squeeze(), '*')
    plt.plot(in_nl, out_nl_approx, 'r')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity output (-)')
    #plt.xlim([-3, 3])
    #plt.ylim([-3, 3])
    plt.grid(True)
