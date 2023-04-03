import matplotlib as mpl

mpl.use("TKAgg")
import torch
import numpy as np
import matplotlib.pyplot as plt
import control

if __name__ == '__main__':
    font = {'size': 14,
            'family': 'serif'}

    mpl.rc('font', **font)
    #mpl.rc('text', usetex=True) probably better for latex compatibility?

    fs = 51200
    ts = 1 / fs

    # Idealized models of G1 and G2 #
    import scipy

    b1, a1 = scipy.signal.cheby1(N=3, rp=0.5, Wn=4.4e3, btype='low', analog=False, output='ba', fs=fs)
    G1 = control.TransferFunction(b1, a1, ts)

    # %%

    plt.figure()
    control.bode(G1, omega_limits=[1e3, 1e5], Hz=True, label="$G_1(z)$", wrap_phase=True, dB=True)

    b2, a2 = scipy.signal.cheby2(N=3, rs=40, Wn=5e3, btype='low', analog=False, output='ba', fs=fs)
    G2 = control.TransferFunction(b2, a2, ts)
    control.bode(G2, omega_limits=[1e3, 1e5], Hz=True, label="$G_2(z)$", wrap_phase=True, dB=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wh_bode.pdf")

    # Idealized model of f(.) #
    y1_lin_min = -5.0  # np.min(y1_lin)
    y1_lin_max = 5.0  # np.max(y1_lin)
    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max - y1_lin_min) / 1000).astype(np.float32).reshape(-1, 1)
    out_nl = -torch.nn.functional.elu(-torch.from_numpy(10 / 11 * (in_nl - 0.0)), alpha=1.0) + 0.0

    font = {'size': 18,
            'family': 'serif'}
    mpl.rc('font', **font)
    plt.figure()
    plt.plot(in_nl, out_nl, 'r')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity output (-)')
    plt.xlim([-3, 3])
    plt.ylim([-3, 1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wh_static.pdf")

    # Bode plot (magnitude only)
    mag1, phase1, omega1 = control.bode_plot(G1, omega_limits=[3e3, 1e5], plot=False)
    mag2, phase2, omega2 = control.bode_plot(G2, omega_limits=[3e3, 1e5], plot=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogx(omega1/(2. * np.pi), 20 * np.log10(mag1), label="$G_1(z)$", linewidth=2)
    ax.semilogx(omega2/(2. * np.pi), 20 * np.log10(mag2), label="$G_2(z)$", linewidth=2)
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("wh_bode_mag.pdf")
