import torch
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import scipy
import numpy as np


def p_tilde(x):
    return torch.exp(-x ** 2 / 2) * torch.sigmoid(10 * x + 4)
    #return torch.exp(-x ** 2 / 2) * torch.sigmoid(10 * x + 4) + 0.8*torch.exp(-(x-3) ** 2 * 10)


def log_p_tilde(x):
    return torch.log(p_tilde(x))
    #return -x ** 2 / 2 + torch.log(torch.sigmoid(10 * x + 4))


if __name__ == "__main__":
    theta_vec = torch.linspace(-2, 4, 10000)
    #theta_vec = torch.linspace(-4, 8, 10000)
    dtheta = theta_vec[1] - theta_vec[0]
    p_theta_tilde = p_tilde(theta_vec)
    Z = p_theta_tilde.sum() * dtheta

    p = lambda theta: p_tilde(theta) / Z
    log_p = lambda theta: log_p_tilde(theta) - torch.log(Z)

    p_x = p_theta_tilde / Z

    log_p_x = p_x.log()

    theta_map = theta_vec[torch.argmax(p_x)]  # simple map estimate

    hess_lap = torch.autograd.functional.hessian(lambda theta: -log_p(theta), theta_map)
    var_lap = 1 / hess_lap
    std_lap = torch.sqrt(var_lap)
    # %%
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 4))
    ax[0].set_title(r"Posterior $p(\theta|\mathcal{D})$")
    ax[0].plot(theta_vec, p(theta_vec), label="Exact")
    ax[0].plot(theta_vec, scipy.stats.norm.pdf(theta_vec, loc=theta_map, scale=std_lap), label="Laplace")
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_xlim([-2, 4])
    #ax[0].axvline(theta_map, color="black", label=r"$\theta^{\rm MAP}$")
    ax[0].plot(theta_map, p(theta_map), "k*", label=r"$\theta^{\rm MAP}$")
    ax[0].legend(loc="upper right")
    #ax[0].grid()

    #ax[0].grid()
    ax[1].set_title(r"Negative log-posterior $-\log p(\theta|\mathcal{D})$")
    ax[1].plot(theta_vec, -log_p(theta_vec), label="Exact")
    ax[1].plot(theta_vec, -log_p(theta_map) + 1 / 2 * hess_lap * (theta_vec-theta_map) ** 2, label="Laplace")
    ax[1].set_xlabel(r"$\theta$")
    ax[1].set_xlim([-2, 4])
    #ax[1].axvline(theta_map, color="black", label=r"$\theta^{\rm MAP}$")
    ax[1].plot(theta_map, -log_p(theta_map), "k*", label=r"$\theta^{\rm MAP}$")
    ax[1].legend(loc="upper right")
    #ax[1].grid()
    plt.savefig("laplace_approx.pdf")


    #ax[1].plot(x, -log_p(x)) # unscaled
    plt.show()
