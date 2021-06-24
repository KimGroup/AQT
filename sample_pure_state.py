from povm import *
from fidelity import *
from time import time

path = 'dicke_6_2'

Nq = 6

Ns = 72900


povm = POVM('pauli6')
Na = povm.Na

# # Load saved state
# st = np.load('{}/{}_st.npy'.format(path, Nq))
# st_sampler = SamplePureState(Nq, povm, st)


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
	pt[basestr2int(data[i], Na)] += 1/Ns

np.save('{}/{}_{}_pt.npy'.format(path, Nq, Ns), pt)

# Reconstruct DM directly from data
dm8 = np.zeros((8, 2**Nq, 2**Nq), dtype=complex)

for xyz in range(8):
	print(xyz)
	dm8[xyz] = GetDMFull(pt, Nq, POVM('pauli6', xyz))

_, negeig, dm = GetBestDM(dm8)
np.save('{}/{}_{}_dm.npy'.format(path, Nq, Ns), dm)