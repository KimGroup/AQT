import time
from fidelity import *
from povm import *
from ann import *
import sys


# Basic parameters

def AQT(datapath, Nq, Nep, Nl=2, dmodel=64, Nh=4, save_model=True, save_loss=True, save_pt=True, save_dm=True):

    povm = POVM('pauli6')
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


    model = InitializeModel(Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=Na).to(device)

    t = time.time()
    model, loss = TrainModel(model, traindata, testdata, device, batch_size=50,lr=1e-4,Nep=Nep)
    print('Took %f minutes'%((time.time()-t)/60))

    model.to('cpu')
    if save_model:
        torch.save(model, '{}.mod'.format(model_filetag))
    if save_loss:
        np.save('{}_loss.npy'.format(model_filetag), loss)

    # Build POVM probability table

    pt_model = POVMProbTable(model)

    if save_pt:
        np.save('{}_pt.npy'.format(model_filetag), pt_model)

    # Reconstruct density matrix

    dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

    for xyz in range(8):
        dm8[xyz] = GetDMFull(pt_model, Nq, POVM('pauli6', xyz))


    _, negeig, dm_model = GetBestDM(dm8)

    if save_dm:
        np.save('{}_dm.npy'.format(model_filetag), dm_model)

    return dm_model

if __name__ == '__main__':

    datapath = 'dicke_6_2/6_72900'
    Nq = 6
    Nep = 100
    
    AQT(datapath, Nq, Nep)