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


# Data
Ns = 40000
data_fname = 'data/ghz-sim-Pauli6-Nq%i-Ns%i.npy'%(Nq, Ns)
data = np.load(data_fname)

traindata = data[:int(len(data)*split)]
testdata = data[int(len(data)*split):]

# Initialize
print('Using cuda:%i'%(int(sys.argv[1])))
device = torch.device("cuda:%i"%(int(sys.argv[1])))

Neplist = np.array([100, 200, 300, 400, 500, 600])
Nlayer, dmodel, Nh = 2, 64, 4

tag = 'aqt-ghz-large'

for epi in range(len(Neplist)):
    Nep = Neplist[epi]
    print('Training with %i epochs'%Nep)
    model = InitializeModel(Nq, Nlayer=Nlayer, dmodel=dmodel, Nh=Nh, Na=Na, dropout=0.0).to(device)

    t = time.time()
    model, losses = TrainModel(model, traindata, testdata, device, smoothing=0.0, batch_size=100,lr=1e-4,Nep=Nep)
    print('Took %f minutes'%((time.time()-t)/60))

    model.to('cpu')

    # torch.save(model, 'mod/%s-%i_%i-%i-%i_%i-%i.mod'%(tag, Nep, Nlayer, dmodel, Nh, Nq, Ns))
