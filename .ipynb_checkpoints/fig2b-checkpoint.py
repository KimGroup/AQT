from datetime import datetime
import time
import h5py
from os import path

from qiskit.quantum_info import state_fidelity


from fidelity import *
from povm import *
from ann import *

now = datetime.now()
datestr = now.strftime("%y%m%d")

# Hyperparameters
Nq = 3
Ns = 3000

povm = POVM('pauli6')
Na = povm.Na

Nlayer, dmodel, Nh = 2, 64, 4
Nep = 400

plist = np.array([0, 0.1, 0.2, 0.3])



tag = 'final-errorscan'


Ne = 4
rhoKB = np.zeros((Ne, 2*Nq), dtype=int)
rhoAmp = np.zeros((Ne), dtype=complex)

# Pure GHZ state
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

dm_ghz = Struct2DM(density)

Nt = 10

qfe = np.zeros((Nt, len(plist)))
qfp = np.zeros((Nt, len(plist)))

for nt in range(Nt):
    for pi in range(len(plist)):

        p = plist[pi]

        # error GHZ state density matrix
        Ne = 8
        rhoKB = np.zeros((Ne, 2*Nq), dtype=int)
        rhoAmp = np.zeros((Ne), dtype=complex)

        # Pure GHZ state
        rhoKB[1, :] = 1
        rhoKB[2, :Nq] = 1
        rhoKB[3, Nq:] = 1

        rhoAmp[0] = 0.5*(1-p)
        rhoAmp[1] = 0.5*(1-p)
        rhoAmp[2] = 0.5*(1-p)
        rhoAmp[3] = np.conj(rhoAmp[2])


        # Error GHZ state
        rhoKB[5, :] = 1
        rhoKB[6, :Nq] = 1
        rhoKB[7, Nq:] = 1
        rhoKB[4:8, 0] = 1-rhoKB[4:8, 0]
        rhoKB[4:8, Nq] = 1-rhoKB[4:8, Nq]

        rhoAmp[4] = 0.5*p
        rhoAmp[5] = 0.5*p
        rhoAmp[6] = 0.5*p
        rhoAmp[7] = np.conj(rhoAmp[6])

        density = {}
        density['KB'] = rhoKB
        density['Amp'] = rhoAmp

        dm_mixed = Struct2DM(density)


        ghz = SampleDM(Nq, povm, density)

        # generate data
        split = 0.9
        data = ghz.samples(Ns=Ns)
        traindata = data[:int(len(data)*split)]
        testdata = data[int(len(data)*split):]

        # training
        device = torch.device("cuda:0")

        model = InitializeModel(Nq, Nlayer=Nlayer, dmodel=dmodel, Nh=Nh, Na=6, dropout=0.0).to(device)


        t = time.time()
        model, losses = TrainModel(model, traindata, testdata, device, smoothing=0.0, batch_size=100,lr=1e-4,Nep=Nep)
        print('Model training took %f minutes'%((time.time()-t)/60))

        model.to('cpu')

        # torch.save(model, 'mod/%s-%i_%i-%i-%i_%i-%i.mod'%(tag, 10*p, Nlayer, dmodel, Nh, Nq, Ns))

        ptab_model = POVMProbTable(model)
        dm_model = GetDMFull(ptab_model, Nq, POVM('pauli6'))

        qfe[nt, pi] = state_fidelity(dm_model, dm_mixed)
        qfp[nt, pi] = state_fidelity(dm_model, dm_ghz)

fname = 'errormodel/%s-qf_%i-%i-%i_%i-%i.h5'%(tag, Nlayer, dmodel, Nh, Nq, Ns)

if path.isfile(fname):
        fname = 'errormodel/%s-%s_%i-%i-%i_%i-%i.h5'%(tag, now.strftime("%y%m%d-%H%M%S"), Nlayer, dmodel, Nh, Nq, Ns)

with h5py.File(fname, 'w') as f:
    f['p'] = plist
    f['qfe'] = qfe
    f['qfp'] = qfp
