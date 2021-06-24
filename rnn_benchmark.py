from datetime import datetime
import time
import h5py
from os import path

from fidelity import *
from povm import *
from ann import *

now = datetime.now()
datestr = now.strftime("%y%m%d")

# Hyperparameters
Nqlist = np.array([20])
Nslist = np.array([8000])
Nv = 2000

povm = POVM('pauli6')
Na = povm.Na

Nlayer, dmodel, Nh = 2, 64, 4

Neplist = np.array([25, 50, 75, 100])

tag = 'final-epochscan'

for qi in range(len(Nqlist)):
    Nq = Nqlist[qi]
    Ns = Nslist[qi]


    # GHZ state density matrix
    Ne = 4
    rhoKB = np.zeros((Ne, 2*Nq), dtype=int)
    rhoAmp = np.zeros((Ne), dtype=complex)

    rhoKB[1, :] = 1
    rhoKB[2, :Nq] = 1
    rhoKB[3, Nq:] = 1

    rhoAmp[0] = 0.5
    rhoAmp[1] = 0.5
    rhoAmp[2] = 0.5
    rhoAmp[3] = np.conj(rhoAmp[2])


    density = {}
    density['KB'] = rhoKB
    density['Amp'] = rhoAmp


    ghz = SampleDM(Nq, povm, density)


    # Initialize data

    split = 0.9

    dname = 'data/%s_%i-%i.npy'%(tag, Nq, Ns)
    if path.isfile(dname):
        data = np.load(dname)
    else:
        data = ghz.samples(Ns=Ns)
        np.save(dname, data)

    traindata = data[:int(len(data)*split)]
    testdata = data[int(len(data)*split):]


    valdata = ghz.samples(Ns=Nv)

    # Training

    device = torch.device("cuda:0")




    cflist = np.zeros(Neplist.shape)

    t0 = time.time()

    for epi in range(len(Neplist)):
        Nep = Neplist[epi]
        model = InitializeModel(Nq, Nlayer=Nlayer, dmodel=dmodel, Nh=Nh, Na=6, dropout=0.0).to(device)


        t = time.time()
        model, losses = TrainModel(model, traindata, testdata, device, smoothing=0.0, batch_size=50,lr=1e-4,Nep=Nep)
        print('Took %f minutes'%((time.time()-t)/60))

        model.to('cpu')

        torch.save(model, 'mod/%s-%i_%i-%i-%i_%i-%i.mod'%(tag, Nep, Nlayer, dmodel, Nh, Nq, Ns))

        cflist[epi] = ClFidEst(ghz, model, samples=valdata)

    print('%i qubits total time: %f minutes'%(Nq, (time.time()-t0)/60))

    fname = 'data/%s-val_%i-%i-%i_%i-%i.h5'%(tag, Nlayer, dmodel, Nh, Nq, Ns)

    if path.isfile(fname):
        fname = 'data/%s-%s_%i-%i-%i_%i-%i.h5'%(tag, now.strftime("%y%m%d-%H%M%S"), Nlayer, dmodel, Nh, Nq, Ns)

    with h5py.File(fname, 'w') as f:
        f['Nep'] = Neplist
        f['cf'] = cflist