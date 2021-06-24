import numpy as np
from scipy.optimize import minimize

from fidelity import *
from povm import *

def int2basestr(n, b, l=0):
    d = int(n%b)
    if d == n:
        return [0 for _ in range(l-1)] + [d]
    else:
        a = int2basestr(int((n-d)/b), b) + [d]
        return [0 for _ in range(l-len(a))] + a
    
    
def TtoRho(t_params, Nq):
    
    if len(t_params) != 4**Nq:
        print('Incorrect t-length: %i'%(len(t_params)))
        return []

    Tm = np.zeros((2**Nq, 2**Nq), dtype=complex)
    
    t_counter = 0
    for i in range(2**Nq):
        Tm[i, i] = t_params[t_counter]
        t_counter += 1
            
    for i in range(2**Nq):
        for j in range(i):
            Tm[i, j] = t_params[t_counter] + t_params[t_counter+1]*1j
            t_counter += 2
            
    rho = np.matmul(np.conjugate(np.transpose(Tm)), Tm)
    
    rho = rho/np.trace(rho)
    
    
    return rho

def MLELoss(t_params, p, Nq, povm):
    Na = povm.Na
    M = povm.M

#     L = 0.
#     rho = TtoRho(t_params)
#     for i in range(d):
#         aa = int2basestr(i, Na, Nq)
#         Mtensor = np.array([1], dtype=complex)
#         for ai in range(Nq):
#             Mtensor = np.kron(Mtensor, M[aa[ai]])
#         q = (np.trace(np.matmul(rho, Mtensor))).real
        
#         L += (q-p[i])**2/q
        
    rho = np.zeros((1, 2**Nq, 2**Nq), dtype=complex)
    rho[0] = TtoRho(t_params, Nq)
    
    
    for nq in range(Nq):
        rho_reduced = np.zeros((Na*len(rho), int(len(rho[0])/2), int(len(rho[0])/2)), dtype=complex)
        for i in range(len(rho)):
            for na in range(Na):
                rho_reduced[Na*i+na] = rho[i, 0::2, 0::2]*M[na, 0, 0]+rho[i, 1::2, 0::2]*M[na, 0, 1]+rho[i, 0::2, 1::2]*M[na, 1, 0]+rho[i, 1::2, 1::2]*M[na, 1, 1]
        rho = rho_reduced
        
    q = rho.flatten().real
    L = np.sum((q-p)**2/q)
    
    return L

def MLE_DM(ptab, Nq, povm):
    Na = povm.Na

    x0 = np.ones(4**Nq)/4**Nq

    t = time.time()
    res = minimize(MLELoss, x0=x0, args=(ptab, Nq, povm), constraints=({'type':'eq', 'fun': lambda x: np.sum(x)-1}))


    dm = TtoRho(res.x, Nq)

    return dm