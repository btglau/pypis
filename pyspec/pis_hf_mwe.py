# -*- coding: utf-8 -*-
"""
May 5 2018

HF MWE

@author: Bryan Lau
"""

import os, glob, argparse
import numpy as np
import scipy.io as sio
import scipy.constants as sc
from scipy import special, integrate

def getArgs():
    # get args from command line sys.argv[1:]
    parser = argparse.ArgumentParser(description="Do a pyscf calculation on a basis set")
    parser.add_argument('-l',type=int,help="basis set size to use",default=4)
    parser.add_argument('-C',type=int,help="basis set size for CB",default=0)
    parser.add_argument('-V',type=int,help="basis set size for VB",default=0)
    parser.add_argument('-n',type=int,help="number of electrons",default=1)
    parser.add_argument('-d',type=float,help="dielectric constant",default=1)
    parser.add_argument('-e',type=float,help='''number of excitations (1 = ground state only, 
                                                                     0 = as many as the basis set size,
                                                                     (0,1) fraction of basis size'''
                        ,default=1)
    parser.add_argument('-r',type=float,help="radius (nm)",default=1)
    parser.add_argument('-s',type=float,help="effective mass (m^(s)tar)",default=1)
    parser.add_argument('-j',help="file name to save results",default=0)
    parser.add_argument('-T',
                        help='''
                        levels of theory to use: give an unordered string with options 
                        needed (HF always done):
                            - (U)HF,
                            - (R)PA (A/B w/o ex),
                            - TD(A) (A/- w/o ex),
                            - (T)DHF (A/B w/ ex),
                            - (C)IS (A/- w/o ex),
                            - CIS(D),
                            - (F)CI,
                            - (E)OM-CCSD')
                        '''
                        ,default='')
    return parser.parse_args()
    #parser.add_argument("fname", help="basis set file name (../basissets/[name])")
    #basisset_path = os.path.normpath('../basissets/' + args.fname)

def change_basis_2el_complex(g,C):
    """Change basis for 2-el integrals with complex coefficients and return.

    - C is a matrix (Ns x Nnew) whose columns are new basis vectors,
      expressed in the basis in which g is given. C is the transformation matrix.
    - Typical operation is g in the site basis, and C is a 
      transformation from site to some-other-basis.
    """
    g1 = np.tensordot(C,g,axes=[0,3])
    g1 = np.transpose(g1,(1,2,3,0))
    # g1 is Ns x Ns x Ns x Nnew
    g = np.tensordot(C,g1,axes=[0,2])
    g = np.transpose(g,(1,2,0,3))
    # g is Ns x Ns x Nnew x Nnew
    g1 = np.tensordot(C.conj(),g,axes=[0,1])
    g1 = np.transpose(g1,(1,0,2,3))
    # g1 is Ns x Nnew x Nnew x Nnew
    g = np.tensordot(C.conj(),g1,axes=[0,0])
    # g is Nnew x Nnew x Nnew x Nnew
    return g

def pis_ao(lmax):
    '''
    return a sequence of numbers that define the PIS basis set for a given lmax
    '''
    assert lmax < 55, 'lmax > 55 means that not all states lower than (n=1),lmax will be returned'
    
    # rows are l, cols are n
    sphjz = np.loadtxt('sphjz_l54_n20.txt',delimiter=',')
    ln = np.transpose(np.nonzero(sphjz**2 <= sphjz[lmax,0]**2))
    ln = ln.astype(np.int_)
    # sort by n first, then l
    ln = ln[np.lexsort((ln[:,1],ln[:,0]))]
    # m_l degeneracy
    aosz = (2*ln[:,0]+1).sum()
    nlm = np.zeros((aosz,3),dtype=np.int_)
    nlm[:,1] = np.repeat(ln[:,0],2*ln[:,0]+1) # l
    # repeat n taking into account m_l degeneracy, and insert m_l
    um = np.unique(2*ln[:,0]+1,return_counts=True)
    ul = np.unique(ln[:,0],return_counts=True)
    ind = np.r_[0,np.array(um).prod(axis=0).cumsum()]
    enumind = np.ndenumerate(ind)
    next(enumind)
    for ix,x in enumind:
        ix = ix[0]-1
        nlm[ind[ix]:x,0] = np.tile(ln[2*ln[:,0]+1==um[0][ix],1],um[0][ix])
        nlm[ind[ix]:x,2] = np.repeat(np.r_[-ul[0][ix]:ul[0][ix]+1],ul[1][ix])
    # jn zeros
    kln = sphjz[nlm[:,1],nlm[:,0]]
    # energy (unitless)
    # eln = kln**2
    # normalization contants
    Nln = np.sqrt(2) / np.absolute(special.spherical_jn(nlm[:,1]+1,kln))
    # from index n to quantum n
    nlm[:,0] += 1
    
    return nlm[:,0],nlm[:,1:],kln,Nln

def dip_mo(R,lmax,MO=None):
    '''
    calculate the dipole matrix elements between AOs, transform them to XYZ 
    cartesian form, then calculate the MO dipole matrix elements
    
    MO - pyscf mo_coeff ndarray - NxN with each column coefficients of MOs
    '''
    lm,kln,Nln = pis_ao(lmax)[1:]
    dsz = kln.size
    
    # preallocate
    dpq = np.zeros((3,dsz,dsz))
    dcart = np.zeros_like(dpq,dtype=np.complex_)
    
    # AO dipole matrix elements
    for i in range(dsz):
        for j in range(dsz):
            dpq[:,i,j] = dip_ao(R,lm[[i,j],0],lm[[i,j],1],kln[[i,j]],Nln[[i,j]])
    
    # cartesian
    dcart[2,...] = dpq[0,...] # z
    dcart[0,...] = dpq[2,...] - dpq[1,...] # x
    dcart[1,...] = 1j*(dpq[1,...] + dpq[2,...]) # y
    dcart[[0,1],...] /= np.sqrt(2)
    
    # mo
    if MO is not None:
        dmo = np.zeros_like(dcart)
        for i in range(dsz):
            for j in range(dsz):
                #dmo[i,j,:] = (np.outer(MO[:,i],MO[:,j])[...,None]*dcart).sum(axis=(0,1))
                # more compact & faster (matmul @) way
                dmo[:,i,j] = MO[:,i].conj().T@dcart@MO[:,j]
        # possible einsum: ab,rbd,cd->rac
        #dmo = np.einsum('ab,rbd,cd->rac',MO.conj(),dcart,MO)
    else:
        dmo = np.zeros(1)
    
    return dpq,dcart,dmo

def dip_ao(R,l,m,kln,Nln):
    '''
    calculate the dipole matrix element between two PIS wavefunctions, with a
    radial bessel function part, and spherical harmonic angular part
    1) angular momentum selection rules
    2) angular part with clebsch-gordon coefficients
    3) radial part numerically integrated
    '''
    dip = np.zeros(3)
    Ym = np.array([0,1,-1])
    
    if np.absolute(m[0]-m[1]) <= 1 and np.absolute(l[0]-l[1]) == 1:
        # constants
        dip += np.sqrt((2*l[1]+1)/(2*l[0]+1))
        dip *= cgc(l[0],l[1],0,0,0)
        # z, x, y angular
        dip[0] *= cgc(l[0],l[1],Ym[0],m[0],m[1])
        dip[1] *= cgc(l[0],l[1],Ym[1],m[0],m[1])
        dip[2] *= cgc(l[0],l[1],Ym[2],m[0],m[1])
        # radial (discard the error estimate)
        dip *= R*integrate.quad(RR,0,1,args=(Nln,kln,l))[0]
        
    return dip

def RR(r,Nln,kln,l):
    '''kernel of radial integral of two bessel functions'''
    return Nln[0]*Nln[1]*np.power(r,3)*\
            special.spherical_jn(l[0],kln[0]*r)*\
            special.spherical_jn(l[1],kln[1]*r) 

def cgc(ll,lr,Ym,ml,mr):
    '''
    a function for evaluating the clebsch-gordon coefficients for the special 
    case of selection rules \Delta l=\pm 1 and \Delta m = 0, \pm 1
    (l)eft, (r)ight
    '''
    c = 0
    if ml-mr == Ym:
        if ll-lr == 1: # \Delta l = 1 already checked
            if Ym == 1:
                c = np.sqrt((lr+ml)*(lr+ml+1)/((2*lr+1)*(2*lr+2)))
            elif Ym == 0:
                c = np.sqrt((lr-ml+1)*(lr+ml+1)/((2*lr+1)*(lr+1)))
            elif Ym == -1:
                c = np.sqrt((lr-ml)*(lr-ml+1)/((2*lr+1)*(2*lr+2)))
        else:
            if Ym == 1:
                c = np.sqrt((lr-ml)*(lr-ml+1)/(2*lr*(2*lr+1)))
            elif Ym == 0:
                c = -np.sqrt((lr-ml)*(lr+ml)/(lr*(2*lr+1)))
            elif Ym == -1:
                c = np.sqrt((lr+ml+1)*(lr+ml)/(2*lr*(2*lr+1)))
    
    return c

def sd(bra,ket,h1,MO):
    '''
    One electron operator matrix element between two slater determinants
    <SD1|h(1)|SD2>
    
    To make use of matmul, please make sure that the h1 operator has the last
    two indices corresponding to the atomic orbital matrix elements
    '''
    orb_diff = bra-ket
    if np.count_nonzero(orb_diff) == 2:
        #matelem = (np.outer(MO[:,orb_diff<0].conj(),MO[:,orb_diff>0])*h1).sum(axis=(0,1))
        matelem = MO[:,orb_diff>0].conj().T@h1@MO[:,orb_diff<0]
    elif np.count_nonzero(orb_diff) == 0:
        matelem = np.zeros(h1.shape[0])
        for i in np.nditer(bra.nonzero()[0]):
            #matelem += bra[i]*(np.outer(MO[:,i].conj(),MO[:,i])*h1).sum(axis=(0,1))
            matelem += MO[:,i].conj().T@h1@MO[:,i]
        # closed shell
        matelem *= 2
        
    return matelem

def spec_ao(R,lmax,n):
    '''
    calculate stick spectrum for AO / basis functions. the transition energies
    are unscaled.
    
    Parameters
    ----------
    lmax : basis set size
    R : size of QD
    n : number of electrons
    
    Returns
    -------
    out : Arrays of stick spectrum energies and strengths
    '''
    
    lm,kln = pis_ao(lmax)[1:3]
    kln = np.square(kln) # energy
    ind = np.lexsort((lm[:,1],kln))
    # ao occupation 
    ao_occ = np.zeros(kln.size)
    # doubly occupied
    docc = np.floor(n/2).astype(np.int_)
    ao_occ[ind[0:docc]] = 2
    # singly occupied
    if n & 0x1:
        ao_occ[ind[docc]] = 1
    
    dcart = np.square(np.absolute(dip_mo(R,lmax)[1])).sum(axis=0)
    # S_{ik}, E_{ik}
    sik = dcart*ao_occ[:,None] # transitions
    sik = sik*(ao_occ[None,:] == 0) # cannot transition to filled orbitals
    eik = kln[None,:] - kln[:,None]
    eik = eik[sik.nonzero()]
    sik = sik[sik.nonzero()]
    
    return sik,eik

def spec_hf(R,lmax,mf):
    '''
    Calculate the stick spectrum for all singles excitations in a HF orbital basis
    
    Parameters
    ----------
    mf : pyscf scf object
    '''
    nmo = mf.mo_coeff.shape[1]
    nocc = np.count_nonzero(mf.mo_occ)
    nvir = nmo - nocc    
    
    sik = np.zeros((nocc,nvir))
    eik = np.zeros_like(sik)
    # matrix elements between MOs
    dmo = np.square(np.absolute(dip_mo(R,lmax,mf.mo_coeff)[2])).sum(axis=0)
    
    for o in range(nocc):
        for v in range(nvir):
            # because <Y_o^v|h1|Y_0> = <v|h1|o>
            # columns are ket, but sik and eik will be flattened anyways
            sik[o,v] = dmo[v+nocc,o]
            eik[o,v] = mf.mo_energy[v+nocc] - mf.mo_energy[o]
    # factor of 2 for a->a and b->b spins
    # Szabo 2.3.5
    sik *= 2
    eik = eik[sik.nonzero()]
    sik = sik[sik.nonzero()]
    
    return sik.ravel(),eik.ravel()

def spec_singles(R,lmax,td,mf):
    '''
    Singles excitation based on a xy matrix, from a tddft pyscf object
    
    Parameters
    ----------
    td : pyscf tddft object
    mf : pyscf scf object (for ground state)
    '''
    
    sik = np.zeros(len(td.xy))
    eik = td.e
    # matrix elements between MOs
    MO = mf.mo_coeff
    dmo = dip_mo(R,lmax,MO)[2] # 3xNxN
    for ind,(x,y) in enumerate(td.xy):
        # shape of x,y in xy matrix is (nvir,nocc), recast it to (None,nvir,nocc)
        nvir,nocc = x.shape
        nmo = nvir + nocc
        sik[ind] = np.square(np.absolute(((x+y)[None,...].conj()*dmo[:,nocc:nmo,0:nocc]))).sum();
    eik = eik[sik > np.finfo(np.float_).eps]
    sik = sik[sik > np.finfo(np.float_).eps]
    sik *= 2
    
    return sik,eik

def unpack_xy(xy):
    '''
    take a td.xy list and unpack it into a dictionary of x and y
    
    Parameters
    ----------
    xy : list of x y matrices
    '''
    x = [i[0] for i in xy]
    y = [i[1] for i in xy]
    return {'x':x,'y':y}

def polarizability(mol, mf, td, ao_dipole=None):
    '''
    A/B X/Y stick spectrum
    Adapted from Berkelbach 2018
    '''
    mol.set_common_orig((0.0,0.0,0.0))
    if ao_dipole is None:
        ao_dipole = mol.intor_symmetric('int1e_r', comp=3) 
    occidx = np.where(mf.mo_occ==2)[0] 
    viridx = np.where(mf.mo_occ==0)[0] 
    mo_coeff = mf.mo_coeff 
    orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx] 
    virocc_dip = np.einsum('xaq,qi->xai', np.einsum('pa,xpq->xaq', orbv, ao_dipole), orbo) 
    sik = np.zeros(len(td.xy))
    eik = td.e
    for ind,(x,y) in enumerate(td.xy):
        sik[ind] = np.square(np.absolute(np.einsum('xai,ai->x',virocc_dip, (x+y)))).sum()
    eik = eik[sik.nonzero()]
    sik = sik[sik.nonzero()]
    sik *= 2
    return sik,eik

def init_pis():
    # parse the input
    args = getArgs()
    from pyscf import gto, scf, lib
    
    a0 = sc.physical_constants['Bohr radius'][0]
    args.r0 = args.r*1E-9/a0 # nm -> a0
    E_scale = 1/(2*args.s*np.square(args.r0))
    
    # load the mat file
    l_path = os.path.normpath('../basissets/' + '*l{}*.mat'.format(args.l))
    basisset_path = glob.glob(l_path)[0]
    mat_contents = sio.loadmat(basisset_path,matlab_compatible=True)

    mol = gto.M()
    mol.verbose = 7
    # if nelectrons is None (initial call to function), then it sets it to
    # tot_electrons = (sum of atom charges - charge)
    # nelectrons = total electrons
    mol.nelectron = args.n
    # prevent pHF from crashing, i.e. force it to use provided AOs
    mol.incore_anyway = True
    norb = int(mat_contents['args']['N'].item())
    
    # python 3 style printing
    print('PIS pyscf')
    print('Basis set size: l = {0}, # fn\'s: {1}'.format(args.l,norb))
    print('{0} electrons'.format(args.n))
    print('{0} threads'.format(lib.num_threads()))
    print('{0} memory assigned'.format(mol.max_memory))
    
    # check if open shell or closed shell problem, based on atomic orbitals
    if args.n not in [2,8,18,20,34,40,58,68,90,92,106,132,138,168,186,196,198,232] or 'U' in args.T:
        print('Open shell system')
        closed_shell = False
        mf = scf.UHF(mol)
    else:
        print('Closed shell system')
        closed_shell = True
        mf = scf.RHF(mol)
    
    # set the core, overlap, and eri integrals
    mf.get_hcore = lambda *args: mat_contents['Hcore'] * E_scale
    mf.get_ovlp = lambda *args: mat_contents['ovlp']
    # get rid of extra dim when sio saves
    mf._eri = (mat_contents['eri']/ args.d / r).squeeze()
    mf.init_guess = '1e'
    
    # handle fractional e (e = number of excitations)
    Nln = pis_ao(args.l)[-1]
    if args.e < 1:
        if args.e <= 0:
            args.e = Nln.size
        else:
            args.e = Nln.size*args.e
    args.e = int(args.e)
    
    return mol,mf,closed_shell,args

if __name__ == '__main__':
    from pyscf import tddft
    mol,mf,closed_shell,args = init_pis()
    E_scale = args.E_scale
    r = args.r0
    mf.kernel()
    sik,eik = spec_hf(args.r0,args.l,mf)

    # RPA (A/B w/ exchange) (=TDHF)
    if 'T' in args.T:
        td = tddft.RPA(mf) # RPA means with exchange in PySCF
        td.nstates = args.e
        td.max_cycle = 500
        td.kernel()
        sik,eik = spec_singles(r,args.l,td,mf)
    
    # RPA (A/- w/ exchange) (=CIS) (+TDA)
    if 'C' in args.T:
        td = tddft.TDA(mf)
        td.nstates = args.e
        td.kernel()
        sik,eik = spec_singles(r,args.l,td,mf)
    
    if 'R' in args.T or 'A' in args.T:
        from pyscf.tddft import rhf_slow

    # RPA (A/B w/o exchange) (RPA)
    if 'R' in args.T:
        td = rhf_slow.dRPA(mf) # equivalent to tddft.TDH(mf)
        td.nstates = args.e
        td.max_cycle = 500
        td.kernel()
        sik,eik = spec_singles(r,args.l,td,mf)
    
    # RPA (A/- w/o exchange) (+TDA)
    if 'A' in args.T:
        td = rhf_slow.dTDA(mf)
        td.nstates = args.e
        td.kernel()
        sik,eik = spec_singles(r,args.l,td,mf)