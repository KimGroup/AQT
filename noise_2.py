from aqt import *

# Hyperparameters
Nq = 3
Ns = 3000

Nep = 100

p = 0.3
Nt = 10



for nt in range(Nt):
    datapath = 'noise_2/{:.1f}/{}_{}_{}'.format(p, Nq, Ns, nt)

    AQT(datapath, Nq, Nep)