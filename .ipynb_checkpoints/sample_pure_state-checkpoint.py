import numpy as np
import povm as P
import fidelity as F
from time import time

path = 'ghz_6'

Nq = 6

Ns = 72900


povm = P.POVM('pauli6')
Na = povm.Na

# Create & save state

# st = np.zeros((2**Nq), dtype=complex)
# st[0] = 1/np.sqrt(2)
# st[-1] = 1/np.sqrt(2)

# np.save('{}/{}_st.npy'.format(path, Nq), st)

# dm = P.PureSt2DM(st)
# np.save('{}/{}_dm.npy'.format(path, Nq), dm)


# # Load saved state
# st = np.load('{}/{}_st.npy'.format(path, Nq))
# st_sampler = P.SamplePureState(Nq, povm, st)


# # Sample data
# t = time()
# data = st_sampler.samples(Ns)
# print(time()-t)

# # Save data
# np.save('{}/{}_{}.npy'.format(path, Nq, Ns), data)

# ALTERNATIVELY work with pre-existing data
data = np.load('{}/{}_{}.npy'.format(path, Nq, Ns))

# Build data frequency distribution
pt = np.zeros((Na**Nq))
for i in range(Ns):
   pt[F.basestr2int(data[i], Na)] += 1/Ns

np.save('{}/{}_{}_pt.npy'.format(path, Nq, Ns), pt)

# Reconstruct DM directly from data
dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

for xyz in range(8):
    print(xyz)
    dm8[xyz] = F.GetDMFull(pt, Nq, P.POVM('pauli6', xyz))

_, negeig, dm = F.GetBestDM(dm8)
np.save('{}/{}_{}_dm.npy'.format(path, Nq, Ns), dm)