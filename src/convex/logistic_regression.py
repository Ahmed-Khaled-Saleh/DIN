

import numpy as np
import matplotlib.pyplot as plt

from src.problems import LogisticRegression
from src.optimizers import *
from src.optimizers.utils import generate_mixing_matrix

from src.experiment_utils import run_exp



if __name__ == '__main__':
    n_agent = 80
    m = 407
    dim = 123    

    kappa = 100
    mu = 5e-8

    n_iters = 10

    p = LogisticRegression(n_agent=n_agent, m=m, dim=dim, noise_ratio=0.05, graph_type='er', kappa=kappa, graph_params=0.3, dataset="a9a", gpu=False)
    print(p.n_edges)


    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)

    W, alpha = generate_mixing_matrix(p) # returns the FDLA mixing matrix(Symmetric fastest distributed linear averaging).
    print('alpha = ' + str(alpha))


    eta = 2/(p.L + p.sigma)
    eta_2 = 2 / (p.L + p.sigma)
    eta_1 = 1 / p.L
    n_inner_iters = int(m * 0.05)
    batch_size = int(m / 10)
    batch_size = 10
    n_dgd_iters = 200
    batch_size = 10
    n_sgd_iters = n_iters * 20
    n_svrg_iters = n_iters
    n_inner_iters = 100

    n_dgd_iters = n_iters * 20


    exps = [
        DGD(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W),
        DIN(p, n_iters=n_dgd_iters, x_0=x_0, W=W),
        DGD_tracking(p, n_iters=n_dgd_iters, eta=eta_1 / 15, x_0=x_0, W=W),
        EXTRA(p, n_iters=n_dgd_iters, eta=eta_1 / 5, x_0=x_0, W=W),
        NIDS(p, n_iters=n_dgd_iters, eta=eta_1, x_0=x_0, W=W),
        DSGD(p, n_iters=n_sgd_iters, eta=eta_1, batch_size=batch_size, x_0=x_0, W=W, diminishing_step_size=True),
        D2(p, n_iters=n_sgd_iters, eta=eta_1, batch_size=batch_size, x_0=x_0, W=W)        
        ]

    res = run_exp(exps, name='logistic_regression', n_cpu_processes=4, save=True, plot= True)


    plt.show()
