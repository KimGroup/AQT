import time
import numpy as np
import torch

import fidelity as F
import povm as P
import ann as A
import sys


# Basic parameters

def AQT(datapath, Nq, Nep, Nl=2, dmodel=64, Nh=4, save_model=True, save_loss=True, save_pt=True, save_dm=True):

    povm = P.POVM('pauli6')
    Na = povm.Na

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load data

    data = np.load('{}.npy'.format(datapath))
    np.random.shuffle(data)

    split = 0.8
    traindata = data[:int(len(data)*split)]
    testdata = data[int(len(data)*split):]


    # Train model

    model_filetag = '{}_{}-{}-{}-{}'.format(datapath, Nep, Nl, dmodel, Nh)


    model = A.InitializeModel(Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=Na).to(device)

    t = time.time()
    model, loss = A.TrainModel(model, traindata, testdata, device, batch_size=50,lr=1e-4,Nep=Nep)
    print('Took %f minutes'%((time.time()-t)/60))

    model.to('cpu')
    if save_model:
        torch.save(model, '{}.mod'.format(model_filetag))
    if save_loss:
        np.save('{}_loss.npy'.format(model_filetag), loss)

    # Build POVM probability table

    pt_model = F.POVMProbTable(model)

    if save_pt:
        np.save('{}_pt.npy'.format(model_filetag), pt_model)

    # Reconstruct density matrix

    dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

    for xyz in range(8):
        dm8[xyz] = F.GetDMFull(pt_model, Nq, P.POVM('pauli6', xyz))


    _, negeig, dm_model = F.GetBestDM(dm8)

    if save_dm:
        np.save('{}_dm.npy'.format(model_filetag), dm_model)

    return dm_model

if __name__ == '__main__':

    datapath = 'ghz_3_ibmq/3_2700'
    Nq = 3
    Nep = 10
    
    AQT(datapath, Nq, Nep)