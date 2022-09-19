# File copied from the repository https://github.com/GerbenBeintema/deepSI in May 2022
# Below is the original copyright for the code in the repo  https://github.com/GerbenBeintema/deepSI

"""
BSD 3-Clause License

Copyright (c) 2019-2020, Gerben Beintema
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL GERBEN BEINTEMA BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.BSD 3-Clause License

Copyright (c) 2019-2020, Gerben Beintema
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL GERBEN BEINTEMA BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from scipy.fftpack import *

import numpy as np

crest_factor = lambda uk: np.max(np.abs(uk))/np.sqrt(np.mean(uk**2))
duplicate = lambda uk,n: np.concatenate([uk]*n)


def multisine(N_points_per_period, N_periods=1, pmin=1, pmax=21, prule=lambda p: p%2==1 and p%6!=1, par=None, n_crest_factor_optim=1):
    '''A multi-sine geneator with only odd frequences and random phases.
    Paramters
    ---------
    N_points_per_period : int
    N_periods : int
    pmin : int
        The lowest number of sin periods allowed in the signal
    pmax : int
        The hightest number of sin periods allowed in the signal
    prule : lambda function
        A function which is true if the p should be allowed. Often used to only select odd frequences
        By default it will allow p%2==1 and p%6!=1 as: [3, 5, 9, 11, 15, 17, 21, 23, 27, 29, 33, 35, 39, 41, 45...]
    par : list of ints like
        Manual list of sin periods in the signal (note: overwrites prule)
    n_crest_factor_optim : int
        n random trails to mimize the crest factor (max(y)/std(y))
    '''

    assert pmax<N_points_per_period//2
    #crest factor optim:
    if n_crest_factor_optim>1:
        ybest = None
        crest_best = float('inf')
        for _ in range(n_crest_factor_optim):
            uk = multisine(N_points_per_period, N_periods=1, pmax=pmax, pmin=pmin, prule=prule, n_crest_factor_optim=1)
            crest = crest_factor(uk)
            if crest<crest_best:
                ybest = uk
                crest_best = crest
        return duplicate(ybest, N_periods)

    N = N_points_per_period

    uf = np.zeros((N,),dtype=complex)
    for p in range(pmin,pmax) if par==None else par:
        if par==None and not prule(p):
            continue
        uf[p] = np.exp(1j*np.random.uniform(0,np.pi*2))
        uf[N-p] = np.conjugate(uf[p])

    uk = np.real(ifft(uf/2)*N)
    uk /= np.std(uk)

    return duplicate(uk, N_periods)


def filtered_signal(N_points_per_period, N_periods=1, fmax=0.1, q=1, transient_periods=5, rng=None):
    '''Generate a signal from filtered uniform noise where u**(1/q) is returned'''
    from scipy import signal
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    u0 = np.random.normal(size=N_points_per_period) if rng is None else rng.normal(size=N_points_per_period)
    u0 = duplicate(u0,N_periods+transient_periods)
    if fmax>=1:
        u1 = u0
    else:
        u1 = signal.lfilter(*signal.butter(6,fmax),u0) #sixth order Butterworth filter

    u1 = u1[N_points_per_period*transient_periods:]
    u1 /= max(np.max(u1),-np.min(u1)) #standardize
    if q==1:
        return u1
    else:
        return abs(u1)**(1/q)*np.sign(u1) if np.isfinite(q) else np.sign(u1)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt
    u = multisine(1000, 5, pmin=250, pmax=499, prule=lambda p: True)
    plt.plot(u)
