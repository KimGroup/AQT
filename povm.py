import numpy as np
import random, os


def int2basestr(n, b, l=0):
    d = int(n%b)
    if d == n:
        return [0 for _ in range(l-1)] + [d]
    else:
        a = int2basestr(int((n-d)/b), b) + [d]
        return [0 for _ in range(l-len(a))] + a
    
def basestr2int(st, b):
    return sum([st[i] * b**(len(st)-i-1) for i in range(len(st))])

def BitFlip(n, N, pos):
    """Flips bit value at position pos from the left of the length-N bit-representation of n"""
    return (n ^ (1 << (N-1-pos)))
def BitGet(n, N, pos):
    """Gets bit value at position pos from the left of the length-N bit-representation of n"""
    return (n >> (N-1-pos) & 1)



class POVM():
    def __init__(self, povm, xyz=0):
        self.povm = povm
        # POVMs and other operators
        # Pauli matrices



        if self.povm == 'tetra':
            self.Na, self.M, self.TinvM = Tetra()
        elif self.povm == 'pauli4':
            self.Na, self.M, self.TinvM = Pauli4(xyz)
        elif self.povm == 'pauli6':
            self.Na, self.M, self.TinvM = Pauli6(xyz)
        else:
            print('Unknown POVM')
            return




        return

def Tetra():

    Pauli = np.array([[[1, 0],[0, 1]],
                      [[0, 1],[1, 0]],
                      [[0, -1j],[1j, 0]],
                      [[1, 0],[0, -1]]])
    Na = 4
    V = np.array([[1., 0, 0, 1.0],
                [1., 2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0],
                [1., -np.sqrt(2.0)/3.0 ,np.sqrt(2.0/3.0), -1.0/3.0 ],
                [1., -np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0 ]])

    M = np.zeros((Na,2,2),dtype=complex)
    
    
    for na in range(Na):
        v = V[na]
        
        for i in range(len(v)):
            M[na,:,:] += 0.25 * v[i] * Pauli[i]

    TinvM = GetT(M)
    return Na, M, TinvM

def Pauli4(xyz):
    Na = 4

    Vz = np.array([[[1],[0]], [[0],[1]]])
    Vx = (1./np.sqrt(2))*np.array([[[1],[1]], [[1],[-1]]])
    Vy = (1./np.sqrt(2))*np.array([[[1],[1j]], [[1],[-1j]]])


    Mz = np.array([np.matmul(V, np.conj(np.transpose(V))) for V in Vz])
    Mx = np.array([np.matmul(V, np.conj(np.transpose(V))) for V in Vx])
    My = np.array([np.matmul(V, np.conj(np.transpose(V))) for V in Vy])

    M = np.zeros((Na, 2, 2), dtype=complex)

    [xi, yi, zi] = int2basestr(xyz, 2, l=3)

    M[0] = (1./3)*Mz[zi]
    M[1] = (1./3)*Mx[xi]
    M[2] = (1./3)*My[yi]
    M[3] = (1./3)*(Mz[1-zi] + Mx[1-xi] + My[1-yi])

    _, _, TinvM = GetT(M)

    return Na, M, TinvM

def Pauli6(xyz):
    Na = 6

    V0 = np.array([[1],[0]])
    V1 = np.array([[0],[1]])
    Vp = (1./np.sqrt(2))*np.array([[1],[1]])
    Vm = (1./np.sqrt(2))*np.array([[1],[-1]])
    Vr = (1./np.sqrt(2))*np.array([[1],[1j]])
    Vl = (1./np.sqrt(2))*np.array([[1],[-1j]])

    M00 = np.matmul(V0, np.conj(np.transpose(V0)))
    M11 = np.matmul(V1, np.conj(np.transpose(V1)))
    Mpp = np.matmul(Vp, np.conj(np.transpose(Vp)))
    Mmm = np.matmul(Vm, np.conj(np.transpose(Vm)))
    Mrr = np.matmul(Vr, np.conj(np.transpose(Vr)))
    Mll = np.matmul(Vl, np.conj(np.transpose(Vl)))

    M = np.zeros((Na, len(V0), len(V0)), dtype=complex)

    
    M[0] = (1./3)*M00
    M[1] = (1./3)*Mpp
    M[2] = (1./3)*Mrr
    M[3] = (1./3)*M11
    M[4] = (1./3)*Mmm
    M[5] = (1./3)*Mll

    [xi, yi, zi] = int2basestr(xyz, 2, l=3)

    _, _, P4TinvM = Pauli4(xyz)

    TinvM = np.zeros((6, 2, 2), dtype=complex)

    TinvM[3*zi] = P4TinvM[0]
    TinvM[1+3*xi] = P4TinvM[1]
    TinvM[2+3*yi] = P4TinvM[2]
    TinvM[3-3*zi] = P4TinvM[3]
    TinvM[4-3*xi] = P4TinvM[3]
    TinvM[5-3*yi] = P4TinvM[3]
    
    return Na, M, TinvM

def GetT(M):
    Na = len(M)

    T = np.zeros((Na, Na), dtype=complex)
    for na1 in range(Na):
        for na2 in range(Na):
            T[na1, na2] = np.trace(np.matmul(M[na1], M[na2]))
    Tinv = np.linalg.inv(T)

    TinvM = np.zeros(M.shape, dtype=complex)

    for na1 in range(Na):
        for na2 in range(Na):
            TinvM[na1, :, :] += Tinv[na1, na2] * M[na2, :, :]
    return T, Tinv, TinvM

class SampleDM():
    def __init__(self, Nq, povm, density):
        self.Nq = Nq

        self.Na = povm.Na
        self.M = np.array(povm.M)

        rhoKB = density['KB']
        rhoAmp = density['Amp']

        self.Ne = len(rhoKB)
        self.KB = np.zeros((self.Ne, 2, self.Nq), dtype=int)
        self.Pl = np.zeros((self.Ne), dtype=complex)
        

        for ne in range(self.Ne):
            self.KB[ne, 0, :] = rhoKB[ne, :Nq]
            self.KB[ne, 1, :] = rhoKB[ne, Nq:2*Nq]
            self.Pl[ne] = rhoAmp[ne]
            
        self.ProjectorMask = np.ones((self.Nq, self.Ne), dtype=bool)
        for nq in range(self.Nq):
            for ne in range(self.Ne):
                for i in range(nq+1, self.Nq):
                    if self.KB[ne, 0, i] != self.KB[ne, 1, i]:
                        self.ProjectorMask[nq, ne] = False
                        break

        return



    def samples(self, Ns):
        
        outcomes = np.zeros((Ns, self.Nq), dtype=int)
        
        for ns in range(Ns):
            pl = np.array(self.Pl)
            pnorm = 1.
            
            for nq in range(self.Nq):
                mask = self.ProjectorMask[nq]
                kb_masked = self.KB[mask, :, :]
                pl_masked = pl[mask]
                
                outp = np.zeros((self.Na),dtype=float)
                
                for na in range(self.Na):
                    for ne in range(len(kb_masked)):
                        outp[na] += (pl_masked[ne]*self.M[na, kb_masked[ne, 1, nq], kb_masked[ne, 0, nq]]).real
                
                if min(outp) < 0.:
                    print('Error: Negative probabilities ', outp)
                    return

                if abs(sum(outp)/pnorm-1) > 0.001:
                    print('Error: Unexpected probability norm',outp, pnorm)
                    return

                outcomes[ns, nq] = np.argmax(np.random.multinomial(n=1, pvals = outp/pnorm))
                
                pnorm = outp[outcomes[ns, nq]]

                for ne in range(self.Ne):
                    pl[ne] *= self.M[outcomes[ns, nq], self.KB[ne, 1, nq], self.KB[ne, 0, nq]]
                    
        return outcomes

    def sample_testing(self, states, getP=False):
        
        outcomes = states
        Ns = len(states)
        PTensor = np.zeros((Ns, self.Nq, self.Na), dtype=float)
        PList = np.zeros((Ns), dtype=float)
        
        for ns in range(Ns):
            pl = np.array(self.Pl)
            pnorm = 1.
            
            for nq in range(self.Nq):
                mask = self.ProjectorMask[nq]
                kb_masked = self.KB[mask, :, :]
                pl_masked = pl[mask]
                
                outp = np.zeros((self.Na),dtype=float)
                
                for na in range(self.Na):
                    for ne in range(len(kb_masked)):
                        outp[na] += (pl_masked[ne]*self.M[na, kb_masked[ne, 1, nq], kb_masked[ne, 0, nq]]).real

                
                if min(outp) < 0.:
                    print('Error: Negative probabilities ', outp)
                    return

                if abs(sum(outp)/pnorm-1) > 0.001:
                    print('Error: Unexpected probability norm',outp, pnorm)
                    return
                
#                 outcomes[ns, nq] = np.random.choice(range(self.Na), p=outp/pnorm)
                PTensor[ns, nq] = outp/pnorm
                
                pnorm = outp[outcomes[ns, nq]]

                for ne in range(self.Ne):
                    pl[ne] *= self.M[outcomes[ns, nq], self.KB[ne, 1, nq], self.KB[ne, 0, nq]]

            PList[ns] = pnorm
                    
        if getP:
            return PTensor, PList
        else:
            return PTensor

    def p(self, state):
        pl = np.array(self.Pl)

        for nq in range(self.Nq):
            for ne in range(self.Ne):
                pl[ne] *= self.M[state[nq], self.KB[ne, 1, nq], self.KB[ne, 0, nq]]

        return sum(pl.real)


class SamplePureState():
    def __init__(self, Nq, povm, S):
        self.Nq = Nq
        self.Na = povm.Na
        self.M = np.array(povm.M)
        self.S = S
        
        
        return
    
    def samples(self, Ns):
        N = self.Nq
        Na = self.Na
        M = self.M
        S = self.S
        
        if len(S) != 2**N:
            print('Incorrect length of S: %i, expected %i'%(len(S), 2**N))
            return

        outcomes = np.zeros((Ns, N), dtype=int)

        for ns in range(Ns):

            S_ket = np.array(S)
            

            pnorm = 1.
            for nq in range(N):
                p = np.zeros((Na))
                S_ket_next = np.zeros((Na, len(S_ket)), dtype=complex)
                
                for na in range(Na):
                    for s in range(2**N):
                        b = BitGet(s, N, nq)
                        S_ket_next[na, s] += M[na, b, b]*S_ket[s]
                        S_ket_next[na, BitFlip(s, N, nq)] += M[na, b^1, b]*S_ket[s]

                    p[na] = np.dot(np.conj(S), S_ket_next[na]).real

                if abs(sum(p)/pnorm-1) > 0.001:
                    print('Error: Unexpected probability norm')

                outcomes[ns, nq] = np.argmax(np.random.multinomial(n=1, pvals = p/pnorm))
                pnorm = p[outcomes[ns, nq]]
                S_ket = S_ket_next[outcomes[ns, nq]]
        return outcomes
    
    def p(self, outcome):
        N = self.Nq
        Na = self.Na
        M = self.M
        S = self.S
        
        S_ket = np.array(S)
        S_ket_next = np.zeros((Na, len(S_ket)), dtype=complex)

        for nq in range(N):
            a = outcome[nq]
            S_ket_next = np.zeros((len(S_ket)), dtype=complex)
            
            for s in range(2**N):
                b = BitGet(s, N, nq)
                S_ket_next[s] += M[a, b, b]*S_ket[s]
                S_ket_next[BitFlip(s, N, nq)] += M[a, b^1, b]*S_ket[s]

            S_ket = S_ket_next
        
        return np.dot(np.conj(S), S_ket).real



def GetPureDM(kets, amps):
    N = len(kets)
    Nq = len(kets[0])

    KB = np.zeros((N**2, 2*Nq), dtype=int)
    Amp = np.zeros((N**2), dtype=complex)

    for i in range(N):
        for j in range(N):
            KB[i*N+j] = np.concatenate((kets[i], kets[j]))
            Amp[i*N+j] = amps[i]*np.conj(amps[j])

    return KB, Amp

def DM2Struct(dm, Nq):
    KB = []
    Amp = []
    
    for i in range(2**Nq):
        istr = int2basestr(i, 2, l=Nq)
        for j in range(2**Nq):
            jstr = int2basestr(j, 2, l=Nq)
            if abs(dm[i,j]) > 0:
                KB.append(np.concatenate((istr, jstr)))
                Amp.append(dm[i,j])
    KB = np.array(KB)
    Amp = np.array(Amp)
    density = {}
    density['KB'] = KB
    density['Amp'] = Amp
    return density

def Struct2DM(density):
    KB = density['KB']
    Amp = density['Amp']
    
    Nq = int(len(KB[0])/2)
    
    dm = np.zeros((2**Nq, 2**Nq), dtype=complex)
    
    for ne in range(len(KB)):
        i = basestr2int(KB[ne, :Nq], 2)
        j = basestr2int(KB[ne, Nq:], 2)
        dm[i,j] = Amp[ne]
        
    return dm

def PureSt2DM(S):
    Svec = np.array([S], dtype=complex)
    return np.matmul(Svec.T, Svec.conj())

    
def LoadData(Nq, fname):

    if os.path.isfile(fname):
        print('Data found: %s'%fname)

        f = open(fname, 'r')
        flines = f.readlines()

        Ns = len(flines)

        data = np.zeros((Ns, Nq), dtype=int)


        for ns in range(Ns):
            fl = flines[ns].split(', ')
            for nq in range(Nq):
                data[ns, nq] = int(fl[nq])
        f.close()
    else:
        print('Data not found')
        data = []

    return data


def GenerateData(gen, Ns, fname):
    if os.path.isfile(fname):
        print('Loading data: %s'%fname)
        data = np.load(fname)
    else:
        print('Generating data: %s'%fname)
        data = gen.samples(Ns=Ns)
        np.save(fname, data)
    return data