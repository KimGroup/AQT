from povm import *
from time import time

path = 'noise_2'

Nq = 3

Ns = 3000

p = 0.3

Nt = 10

povm = POVM('pauli6')
Na = povm.Na

# Create states

# st0 = np.zeros((2**Nq), dtype=complex)
# st0[0] = 1/np.sqrt(2)
# st0[-1] = 1/np.sqrt(2)
# np.save('{}/{}_st0.npy'.format(path, Nq), st0)

# st1 = np.zeros((2**Nq), dtype=complex)
# st1[basestr2int((1,0,0), 2)] = 1/np.sqrt(2)
# st1[basestr2int((0,1,1), 2)] = 1/np.sqrt(2)
# np.save('{}/{}_st1.npy'.format(path, Nq), st1)

# st2 = np.zeros((2**Nq), dtype=complex)
# st2[basestr2int((0,1,0), 2)] = 1/np.sqrt(2)
# st2[basestr2int((1,0,1), 2)] = 1/np.sqrt(2)
# np.save('{}/{}_st2.npy'.format(path, Nq), st2)

# st3 = np.zeros((2**Nq), dtype=complex)
# st3[basestr2int((0,0,1), 2)] = 1/np.sqrt(2)
# st3[basestr2int((1,1,0), 2)] = 1/np.sqrt(2)
# np.save('{}/{}_st3.npy'.format(path, Nq), st3)


# Load saved state
st0 = np.load('{}/{}_st0.npy'.format(path, Nq))
st0_sampler = SamplePureState(Nq, povm, st0)

st1 = np.load('{}/{}_st1.npy'.format(path, Nq))
st1_sampler = SamplePureState(Nq, povm, st1)

st2 = np.load('{}/{}_st2.npy'.format(path, Nq))
st2_sampler = SamplePureState(Nq, povm, st2)

st3 = np.load('{}/{}_st3.npy'.format(path, Nq))
st3_sampler = SamplePureState(Nq, povm, st3)

# Save density matrix

# dm0 = PureSt2DM(st0)
# dm1 = PureSt2DM(st1)
# dm2 = PureSt2DM(st2)
# dm3 = PureSt2DM(st3)

# dm = (1-p)*dm0 + (p/3)*(dm1+dm2+dm3)

# np.save('{}/{:.1f}/{}_dm.npy'.format(path, p, Nq), dm)



# Sample data
rng = np.random.default_rng()

for nt in range(Nt):
    data = np.zeros((Ns, Nq), dtype=int)
    key = rng.random(Ns)

    for i in range(Ns):
        if key[i] < 1-p:
            data[i] = st0_sampler.samples(1)
        elif key[i] < 1-2*p/3:
            data[i] = st1_sampler.samples(1)
        elif key[i] < 1-p/3:
            data[i] = st2_sampler.samples(1)
        else:
            data[i] = st3_sampler.samples(1)


    # Save data
    np.save('{}/{:.1f}/{}_{}_{}.npy'.format(path, p, Nq, Ns, nt), data)

