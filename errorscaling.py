from datetime import datetime
import time
from os import path
import sys

from fidelity import *
from povm import *
from ann import *

now = datetime.now()
datestr = now.strftime("%y%m%d")


# Hyperparameters
Nq = 6
split = 0.8

povm = POVM('pauli6')
Na = povm.Na

# GHZ state density matrix
# Ne = 4
# rhoKB = np.zeros((Ne, 2*Nq), dtype=int)
# rhoAmp = np.zeros((Ne), dtype=complex)

# rhoKB[1, :] = 1
# rhoKB[2, :Nq] = 1
# rhoKB[3, Nq:] = 1

# rhoAmp[0] = 0.5
# rhoAmp[1] = 0.5
# rhoAmp[2] = 0.5
# rhoAmp[3] = np.conj(rhoAmp[2])


# density = {}
# density['KB'] = rhoKB
# density['Amp'] = rhoAmp


# ghz = SampleDM(Nq, povm, density)


tag = 'errorscaling/ghz-errorscaling'

# Initialize
print('Using cuda:%i'%(int(sys.argv[1])))
device = torch.device("cuda:%i"%(int(sys.argv[1])))

Nlayer, dmodel, Nh = 2, 64, 4


# Data
# Ns = 80000
# Neplist = np.array([100, 200, 400])

Nslist = np.array([40000, 80000, 130000, 200000, 400000])
Nep = 200

for nsi in range(len(Nslist)):
    Ns = Nslist[nsi]
    data_fname = '%s_%i-%i.npy'%(tag, Nq, Ns)
    # data = ghz.samples(Ns=Ns)
    # np.save(data_fname, data)
    data = np.load(data_fname)

    traindata = data[:int(len(data)*split)]
    testdata = data[int(len(data)*split):]

    print('Training with %i epochs'%Nep)
    model = InitializeModel(Nq, Nlayer=Nlayer, dmodel=dmodel, Nh=Nh, Na=Na, dropout=0.0).to(device)

    t = time.time()
    model, losses = TrainModel(model, traindata, testdata, device, smoothing=0.0, batch_size=100,lr=1e-4,Nep=Nep)
    print('Took %f minutes'%((time.time()-t)/60))

    model.to('cpu')

    torch.save(model, '%s-%i_%i-%i-%i_%i-%i.mod'%(tag, Nep, Nlayer, dmodel, Nh, Nq, Ns))
