import time
from fidelity import *
from povm import *
from ann import *



# Basic parameters

path = 'noise_2/0.3'

Nq = 3

Ns = 3000

povm = POVM('pauli6')
Na = povm.Na

# Hyperparameters

Nl, dmodel, Nh = 2, 64, 4
Nep = 100

print(torch.cuda.is_available())
device = torch.device("cuda:0")



# Load data

data = np.load('{}/{}_{}.npy'.format(path, Nq, Ns))
np.random.shuffle(data)

split = 0.8
traindata = data[:int(len(data)*split)]
testdata = data[int(len(data)*split):]


# Train model

model_filetag = '{}/{}_{}_{}-{}-{}-{}'.format(path, Nq, Ns, Nep, Nl, dmodel, Nh)


model = InitializeModel(Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=Na).to(device)

t = time.time()
model, loss = TrainModel(model, traindata, testdata, device, batch_size=50,lr=1e-4,Nep=Nep)
print('Took %f minutes'%((time.time()-t)/60))

model.to('cpu')
torch.save(model, '{}.mod'.format(model_filetag))
np.save('{}_loss.npy'.format(model_filetag), loss)

# Build POVM probability table

pt_model = POVMProbTable(model)

np.save('{}_pt.npy'.format(model_filetag), pt_model)

# Reconstruct density matrix

dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

for xyz in range(8):
    dm8[xyz] = GetDMFull(pt_model, Nq, POVM('pauli6', xyz))


_, negeig, dm_model = GetBestDM(dm8)

np.save('{}_dm.npy'.format(model_filetag), dm_model)

