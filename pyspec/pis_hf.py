# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:13:24 2017

@author: Bryan Lau

1/19/2018 - change CISD to FCI, to get singlet states easily
3/8/2018 - add UHF capability, move from c0, c1, and c2 to rdm
3/16/2018 - switch EE-CCSD part to GF based
3/23/2018 - add python functions for calculating spectra
"""

import os, glob, argparse, h5py
import numpy as np
import scipy.io as sio
from scipy import constants
from scipy.integrate import quad
from scipy.special import spherical_jn
from scipy.linalg import block_diag

'''
upon execution, this program should look up a directory, then into a folder
called 'basissets'. It will load the appropriate basis set size, then do some
combination of HF / pHF. For a single band only.
'''

'''
User-defined Hamiltonian for SCF module.

Three steps to define Hamiltonian for SCF:
1. Specify the number of electrons. (Note mole object must be "built" before doing this step)
2. Overwrite three attributes of scf object
    .get_hcore
    .get_ovlp
    ._eri
3. Specify initial guess (to overwrite the default atomic density initial guess)

Note you will see warning message on the screen:

        overwrite keys get_ovlp get_hcore of <class 'pyscf.scf.hf.RHF'>
'''

def getArgs():
    # get args from command line sys.argv[1:]
    parser = argparse.ArgumentParser(description="Do a pyscf calculation on a basis set")
    parser.add_argument('-l',help="basis set size to use (or comma separated of n)",default=[4])
    parser.add_argument('-C',type=int,help="basis set size for CB",default=0)
    parser.add_argument('-V',type=int,help="basis set size for VB",default=0)
    parser.add_argument('-n',type=int,help="number of electrons",default=2)
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
                        needed:
                            - A(O),
                            - HF (M)ean field,
                            - (U)HF,
                            - (R)PA (A/B w/o ex),
                            - TD(A) (A/- w/o ex),
                            - (T)DHF (A/B w/ ex),
                            - (C)IS (A/- w/ ex),
                            --(H) diagonalizes full H for (T)DHF and (R)PA,
                            - CIS(D),
                            - (F)CI,
                            - (E)OM-CCSD')
                        '''
                        ,default='')
    parser.add_argument('-R',help='Energy range "start,stop,steps" in eV if -T E',default='')
    args = parser.parse_args()
    
    # convert lmax to an array [n of l=0,n of l=1,...] or just lmax = args.l
    args.l_string = args.l
    args.l = np.asarray(args.l.split('-'),dtype='int_')
    if args.l.size == 1:
        args.l = args.l[0]
    
    # handle fractional e (e = number of excitations)
    Nnl = pis_ao(args.l)[-1]
    if args.e < 1:
        norb = Nnl.size
        nvir = norb - args.n/2
        nocc = args.n/2
        num_es = nocc*nvir
        if args.e <= 0:
            args.e = num_es
        else:
            args.e = num_es*args.e
    args.e = int(args.e)
    
    # work up energy range input for (E)OM-CCSD
    if 'E' in args.T:
        args.ccsd_range = np.asarray(args.R.split(',')).astype(np.float)
        
    # nm -> a0
    args.r0 = args.r*1E-9/constants.physical_constants['Bohr radius'][0]
    
    return args

def pis_ao(lmax):
    '''
    return a sequence of numbers that define the PIS basis set for a given lmax
    '''
    sphjz = np.loadtxt('sphjz_l54_n20.txt',delimiter=',')
    if np.isscalar(lmax):
        assert lmax < 55, 'lmax > 55 means that not all states lower than (n=1),lmax will be returned'
        # rows are l, cols are n
        ln = np.transpose(np.nonzero(sphjz**2 <= sphjz[lmax,0]**2))
        ln = ln.astype(np.int_)
    else:
        ln = np.zeros((lmax.sum(),2),dtype=np.int_)
        ln[:,0] = np.repeat(np.r_[0:lmax.size],lmax)
        for l,n in enumerate(lmax): # ind, value
            ln[ln[:,0]==l,1] = np.r_[0:n]
    
    # sort by l first, then n
    ln = ln[np.lexsort((ln[:,1],ln[:,0]))]
    # m_l degeneracy
    aosz = (2*ln[:,0]+1).sum()
    nlm = np.zeros((aosz,3),dtype=np.int)
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
    knl = sphjz[nlm[:,1],nlm[:,0]]
    # energy (unitless)
    # eln = knl**2
    # normalization contants
    Nnl = np.sqrt(2) / np.absolute(spherical_jn(nlm[:,1]+1,knl))
    # from index n to quantum n
    nlm[:,0] += 1
    
    # n [l m] knl Nnl
    return nlm[:,0],nlm[:,1:],knl,Nnl

def dip_mo(R,lmax,MO=None):
    '''
    calculate the dipole matrix elements between AOs, transform them to XYZ 
    cartesian form, then calculate the MO dipole matrix elements
    
    MO - pyscf mo_coeff ndarray - NxN with each column coefficients of MOs
    '''
    lm,knl,Nnl = pis_ao(lmax)[1:]
    dsz = knl.size
    
    # preallocate
    dpq = np.zeros((3,dsz,dsz))
    dcart = np.zeros_like(dpq,dtype=np.complex_)
    
    # AO dipole matrix elements
    for i in range(dsz):
        for j in range(dsz):
            dpq[:,i,j] = dip_ao(R,lm[[i,j],0],lm[[i,j],1],knl[[i,j]],Nnl[[i,j]])
    
    # transform from complex to real
    U = Ulmax(lmax) # <u|m>
    dpq = U @ dpq @ U.conj().T # U^+ D U 
    
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
                # matmal (@) treats n-d array as slices of matrices in the last
                # two indices
                dmo[:,i,j] = MO[:,i].conj().T@dcart@MO[:,j]
    else:
        dmo = np.zeros(1)
    
    return dpq,dcart,dmo

def dip_ao(R,l,m,knl,Nnl):
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
        dip *= R*quad(RR,0,1,args=(Nnl,knl,l))[0]
        
    return dip

def RR(r,Nnl,knl,l):
    '''kernel of radial integral of two bessel functions'''
    return Nnl[0]*Nnl[1]*np.power(r,3)*\
            spherical_jn(l[0],knl[0]*r)*\
            spherical_jn(l[1],knl[1]*r) 

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

def Ulmax(lmax):
    '''
    Return a transformation matrix that follows the AO list given by pis_ao.
    Different from Ulmu that assumes l is the same for a given u and m.
    '''
    n,lm = pis_ao(lmax)[0:2]
    nl,ind = np.unique(np.c_[n,lm[:,0]],return_index=True,axis=0)
    nl = nl[np.argsort(ind)]
    U_new = list()
    if np.isscalar(lmax):
        lmax += 1
    else:
        lmax = lmax.size
    for a in range(0,lmax):
        b = np.nonzero(nl[:,1]==a)
        step = b[0].size
        Us_size = step*(2*a+1)
        Us = np.zeros((Us_size,Us_size),dtype='complex_')
        for c in range(0,step):
            # nest the tranfsormation matrices for a given l in the same pattern
            # as given by pis_ao
            Us[c::step,c::step] = Ulmu(np.arange(-a,a+1)[:,None],np.arange(-a,a+1)[None,:])
        U_new.append(Us)
    U = block_diag(*U_new)
    return U

def Ulmu(u,m):
    '''
    Transformation matrix for complex to real spherical harmonics, <u|m>
    Always returns positive real spherical harmonics
    '''
    u = u.astype('float_')
    m = m.astype('float_')
    U = (m==0)*(u==0) + ((u>0)*np.power(-1,m)*(m==u) \
                         + 1j*(-u>0)*(m==u) \
                         - 1j*(-u>0)*np.power(-1,m)*(m==-u) \
                         + (u>0)*(m==-u))/np.sqrt(2)
    
    return U

def sd(bra,ket,h1,MO):
    '''
    One electron operator matrix element between two slater determinants
    <SD1|h(1)|SD2>
    
    To make use of np/matmul, please make sure that the h1 operator has the last
    two indices corresponding to the atomic orbital matrix elements, while other
    indices to the left can be coordinates, etc.
    '''
    orb_diff = bra-ket
    if np.count_nonzero(orb_diff) == 2:
        matelem = MO[:,orb_diff>0].conj().T@h1@MO[:,orb_diff<0]
    elif np.count_nonzero(orb_diff) == 0:
        matelem = np.zeros(h1.shape[0])
        for i in np.nditer(bra.nonzero()[0]):
            matelem += MO[:,i].conj().T@h1@MO[:,i]
        # closed shell
        matelem *= 2
        
    return matelem

def spec_ao(R,lmax,n):
    '''
    Calculate stick spectrum for unitless AO / basis functions. 
    The transition energies are unscaled.
    
    Parameters
    ----------
    lmax : basis set size
    R : size of QD
    n : number of electrons
    
    Returns
    -------
    out : Arrays of stick spectrum energies and strengths
    
    |Y> = \sum c_i|AO>, where c_i = 0, 1, or 2
    '''
    
    lm,knl = pis_ao(lmax)[1:3]
    knl = np.square(knl) # energy
    ind = np.lexsort((lm[:,1],knl))
    # ao occupation 
    ao_occ = np.zeros(knl.size)
    # doubly occupied
    docc = np.floor(n/2).astype(np.int_)
    ao_occ[ind[0:docc]] = 2
    # singly occupied
    if n & 0x1:
        ao_occ[ind[docc]] = 1
    dcart = dip_mo(R,lmax)[1]
    
    sik = dcart*ao_occ[:,None] # transitions
    sik = sik*(ao_occ[None,:] == 0) # cannot transition to filled orbitals
    # S_{ik}, E_{ik}
    sik = np.square(np.absolute(sik)).sum(axis=0)
    eik = knl[None,:] - knl[:,None]
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
    if hasattr(mf,'dmo'):
        dmo = mf.dmo
    else:
        dmo = dip_mo(R,lmax,mf.mo_coeff)[2]
    
    for o in range(nocc):
        for v in range(nvir):
            # <Y_o^v|h1|Y_0> = <v|h1|o>
            # <Y_0|h1|Y_o^v> = <o|h1|v>
            # factor of 2 for a->a and b->b spins
            # Szabo 2.3.5
            sik[o,v] = np.square(np.absolute(2*dmo[:,o,v+nocc])).sum(axis=0)
            #sik[o,v] = (4*dmo[:,o,v+nocc]*dmo[:,v+nocc,o]).sum(axis=0)
            eik[o,v] = mf.mo_energy[v+nocc] - mf.mo_energy[o]
    eik = eik[sik.nonzero()]
    sik = sik[sik.nonzero()]
    
    return sik.ravel(),eik.ravel()

def spec_singles(R,lmax,mf,td):
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
    if hasattr(mf,'dmo'):
        dmo = mf.dmo
    else:
        dmo = dip_mo(R,lmax,mf.mo_coeff)[2]
        
    for ind,(x,y) in enumerate(td.xy):
        # shape of x,y in xy matrix is (nvir,nocc), recast it to (None,nvir,nocc)
        nvir,nocc = x.shape
        nmo = nvir + nocc
        sik[ind] =  np.square(np.absolute(((x+y).T[None,...]*dmo[:,0:nocc,nocc:nmo]).sum(axis=(1,2)))).sum()
        
    # prune td.xy and td.e to return only non-zero vectors
    ind = sik > np.finfo(np.float_).eps
    td.xy = [td.xy[c] for c in range(len(td.xy)) if ind[c]]
    td.e = [td.e[c] for c in range(len(td.xy)) if ind[c]]
    eik = eik[ind]
    sik = sik[ind]
    sik *= 2
    
    return sik,eik

def polarizability_stick(args,mol,mf,td):
    '''
    A/B X/Y stick spectrum
    Adapted from Berkelbach 2018
    '''
    if hasattr(mf,'dmo'):
        ao_dipole = mf.dcart
    else:
        ao_dipole = dip_mo(args.r0,args.l,mf.mo_coeff)[1]
        
    occidx = np.where(mf.mo_occ==2)[0] 
    viridx = np.where(mf.mo_occ==0)[0] 
    mo_coeff = mf.mo_coeff 
    orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx] 
    virocc_dip = np.einsum('xaq,qi->xai', np.einsum('pa,xpq->xaq', orbv, ao_dipole), orbo) 
    sik = np.zeros(len(td.xy))
    eik = td.e
    for ind,(x,y) in enumerate(td.xy):
        sik[ind] = np.square(np.absolute(np.einsum('xai,ai->x',virocc_dip, (x+y)))).sum()
    # only nonzero
    ind = sik > np.finfo(np.float_).eps
    eik = eik[ind]
    sik = sik[ind]
    sik *= 2
    
    return sik,eik

def polarizability(args,mol,mf,td): 
    '''
    absorption spectrum, Tim's way
    '''
    Ha = constants.physical_constants['Hartree energy in eV'][0]
    ws = args.ccsd_range
    # convert eV -> Ha
    ws[0:2] /= Ha
    omegas = np.linspace(ws[0],ws[1],num=ws[2].astype(np.int))+1j*broaden(args.r)/Ha    
    MO = mf.mo_coeff
    if hasattr(mf,'dmo'):
        ao_dipole = mf.dcart
    else:
        ao_dipole = dip_mo(args.r0,args.l,MO)[2] # 3xNxN
        
    occidx = np.where(mf.mo_occ==2)[0] 
    viridx = np.where(mf.mo_occ==0)[0] 
    mo_coeff = mf.mo_coeff 
    orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx] 
    virocc_dip = np.einsum('xaq,qi->xai', np.einsum('pa,xpq->xaq', orbv, ao_dipole), orbo) 
    p = np.zeros((omegas.size,3), dtype=np.complex128) 
    for (x,y),e in zip(td.xy, td.e): 
        dip = np.einsum('xai,ai->x',virocc_dip, np.sqrt(2.0)*(x+y))
        for iw,w in enumerate(omegas): 
            p[iw] += dip**2*((1.0/(w-e))-(1.0/(w+e))) 
            
    return p,omegas.real

def spec_rdm(R,lmax,trdm,e,mf):
    '''
    Stick spectrum from any set of transition RDMs:
    <0|p^+ q|n> = y_qp
    S_ge = Tr(ry)Tr(r*y*)
    
    Parameters
    ----------
    trdm : list of transition 1 particle density matrices
    mf : pyscf scf object (for ground state)
    '''
    if hasattr(mf,'dmo'):
        dmo = mf.dmo
    else:
        dmo = dip_mo(R,lmax,mf.mo_coeff)[2]
    sik = np.zeros(len(trdm))
    eik = np.zeros_like(sik)
    for ind,t in enumerate(trdm):
        #sik[ind] = np.square(np.absolute(np.trace(dmo@t,axis1=1,axis2=2))).sum()
        sik[ind] = (np.trace(dmo.conj()@t.conj(),axis1=1,axis2=2)*np.trace(dmo@t,axis1=1,axis2=2)).sum()
        eik[ind] = e[ind] - e[0]
    # matrix element for <gs|r|gs> can be nonzero, if gs is not a pure angular
    # momentum state. zero out any entries where eik == 0
    ind = np.logical_or(sik > np.finfo(np.float_).eps,eik==0)
    eik = eik[sik > np.finfo(np.float_).eps]
    sik = sik[sik > np.finfo(np.float_).eps]
    
    return sik,eik

def broaden(r):
    '''
    broadening curve fit to Alan's parameters
    takes r in nm, returns in eV
    '''
    return 0.05487*np.power(r,-1.08)

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

def init_pis():
    from pyscf import gto, scf, lib
    
    # parse the input
    args = getArgs()
    E_scale = 1/(2*args.s*np.square(args.r0))
    
    # load the mat file
    l_path = os.path.normpath('../basissets_8fold/' + '*l{}_*.mat'.format(args.l_string))
    #l_path = os.path.normpath('../basissets/' + '*l{}*.mat'.format(args.l))
    basisset_path = glob.glob(l_path)[0]
    
    # .mat
    #mat_contents = sio.loadmat(basisset_path,matlab_compatible=True)
    # hdf5
    mat_contents = h5py.File(basisset_path,'r')

    mol = gto.M()
    mol.verbose = 7
    # if nelectrons is None (initial call to function), then it sets it to
    # tot_electrons = (sum of atom charges - charge)
    # nelectrons = total electrons
    mol.nelectron = args.n
    # prevent pHF from crashing, i.e. force it to use provided AOs
    mol.incore_anyway = True
    args.norb = int(mat_contents['args']['N'][0])
    
    # python 3 style printing
    print('PIS pyscf')
    print('Basis set size: l = {0}, # fn\'s: {1}'.format(args.l_string,args.norb))
    print('{0} electrons'.format(args.n))
    print('{0} threads'.format(lib.num_threads()))
    print('{0} memory assigned'.format(mol.max_memory))
    print('{0} scratch directory'.format(lib.param.TMPDIR))
    
    # check if open shell or closed shell problem, based on atomic orbitals
    if not closed_shell(args.n) or 'U' in args.T:
        print('Open shell system')
        mf = scf.UHF(mol)
    else:
        print('Closed shell system')
        mf = scf.RHF(mol)
    
    # set the core, overlap, and eri integrals - load hdf5 file
    Hcore = mat_contents['Hcore'][:]*E_scale
    ovlp = mat_contents['ovlp'][:]
    eri = (mat_contents['eri'][:]/args.d/args.r0).squeeze()
    # close
    mat_contents.close()
    mf.get_hcore = lambda *args: Hcore
    mf.get_ovlp = lambda *args: ovlp
    mf._eri = eri
    mf.init_guess = '1e'
    
    return mol,mf,args

def closed_shell(n):
    '''
    Electron filling order generated by ordering PIS wf\'s (n, l, index) by energy,
    then taking the cumulative sum of 2(2l+1) (2 e- per shell + degeneracy)
    
    MATLAB code:
        sphjz = besselzero(((0:lmax)+0.5)',20);
        [l,n] = find(sphjz.^2<=sphjz(end,1).^2);
        l = l-1;
        e = sphjz(find(sphjz.^2<=sphjz(end,1).^2)).^2;
        [nle,ind] = sortrows([n l e],[3 2 1]);
        filling = cumsum(2*(2*nle(:,2)+1));
        
    [1s 1p 1d 2s 1f 2p 1g 2d ...]
    '''
    #return n in [2,8,18,20,34,40,58,68,90,92,106,132,138,168,186,196,198,232]
    #HF ordering changes from AO for n > 2, so this function is no longer valid
    return True
    

def do_ao(args,specout):
    # AO
    E_scale = 1/(2*args.s*np.square(args.r0))
    n,lm,knl,Nnl = pis_ao(args.l)
    sik,eik = spec_ao(args.r0,args.l,args.n)
    specout['E'] = {'AO':np.square(knl)*E_scale}
    specout['C'] = {'AO':{'n':n,'lm':lm,'Nnl':Nnl}}
    specout['S'] = {'AO':{'sik':sik,'eik':eik*E_scale}}
    return specout

def do_hf(mf,args,specout):
    # HF: returns converged, e_tot, mo_energy, mo_coeff, mo_occ
    mf.kernel()
    # attach d_pq to mf object so recalculation in post-HF is not neccessary
    mf.dcart,mf.dmo = dip_mo(args.r0,args.l,MO=mf.mo_coeff)[1:]
    if 'M' in args.T:
        specout['E'].update({'HF':{'e_tot':mf.e_tot,'mo_energy':mf.mo_energy}})
        specout['C'].update({'HF':{'mo_coeff':mf.mo_coeff,'mo_occ':mf.mo_occ}})
        specout['conv'].update({'HF':mf.converged})
        sik,eik = spec_hf(args.r0,args.l,mf)
        specout['S'].update({'HF':{'sik':sik,'eik':eik}})
        
    return specout,mf

def do_singles(mf,args,specout):
    '''
    sik,eik = polarizability(args,mol,mf,td,ao_dipole=None)
    '''
    from pyscf import tddft
    import rhf_slow, pis_ab_direct
    r = args.r0
    lmax = args.l
        
    # (A/B w/ exchange) (=TDHF/RPA)
    if 'T' in args.T:
        td = None
        if 'H' in args.T:
            td = pis_ab_direct.direct_TDHF(mf)
            td.converged = True
        else:
            td = rhf_slow.TDHF(mf) # RPA means with exchange in PySCF
            td.nstates = args.e
            #td.max_cycle = 500
            td.kernel()
        sik,eik = spec_singles(r,lmax,mf,td)
        specout['E'].update({'TDHF':td.e})
        specout['C'].update({'TDHF':unpack_xy(td.xy)})
        specout['conv'].update({'TDHF':td.converged})
        specout['S'].update({'TDHF':{'sik':sik,'eik':eik}})
        #td.e td.xy td.converged
    
    # (A/- w/ exchange) (=CIS) (+TDA)
    if 'C' in args.T:
        td = None
        td = tddft.TDA(mf)
        td.nstates = args.e
        td.kernel()
        sik,eik = spec_singles(r,lmax,mf,td)
        specout['E'].update({'CIS':td.e})
        specout['C'].update({'CIS':unpack_xy(td.xy)})
        specout['conv'].update({'CIS':td.converged})
        specout['S'].update({'CIS':{'sik':sik,'eik':eik}})

    # (A/B w/o exchange) (direct RPA), i.e. TDH
    if 'R' in args.T:
        td = None
        if 'H' in args.T:
            td = pis_ab_direct.direct_dRPA(mf)
            td.converged = True
        else:
            td = rhf_slow.dRPA(mf) # equivalent to tddft.TDH(mf)
            td.nstates = args.e
            #td.max_cycle = 500
            td.kernel()
        sik,eik = spec_singles(r,lmax,mf,td)
        specout['E'].update({'RPA':td.e})
        specout['C'].update({'RPA':unpack_xy(td.xy)})
        specout['conv'].update({'RPA':td.converged})
        specout['S'].update({'RPA':{'sik':sik,'eik':eik}})
    
    # (A/- w/o exchange) (direct TDA), i.e TDH-TDA
    if 'A' in args.T:
        td = None
        td = rhf_slow.dTDA(mf)
        td.nstates = args.e
        td.kernel()
        sik,eik = spec_singles(r,lmax,mf,td)
        specout['E'].update({'TDA':td.e})
        specout['C'].update({'TDA':unpack_xy(td.xy)})
        specout['conv'].update({'TDA':td.converged})
        specout['S'].update({'TDA':{'sik':sik,'eik':eik}})
    
    return specout

def do_cisd(mol,mf,args,specout):
    from pyscf import ci,fci

    norb = args.norb
    nelec = mol.nelec
    myci = ci.CISD(mf)
    myci.nroots = args.e
    myci.max_cycle = 500
    
    nmo = mf.mo_coeff.shape[1]
    nocc = np.count_nonzero(mf.mo_occ)
    nvir = nmo - nocc 
    ci_init = []
    c0 = 0
    c2 = np.zeros((nocc,nocc,nvir,nvir))
    for i in range(myci.nroots):
        c1 = np.zeros(nocc*nvir)
        c1[i] = 1
        c1 = c1.reshape(nocc,nvir)
        ci_init.append(myci.amplitudes_to_cisdvec(c0, c1, c2))
    
    myci.kernel(ci0=ci_init)
    
    CISD_C = list()
    if not closed_shell(args.n):
        norbci = myci.get_nmo()
    else:
        norbci = norb
    # get the ground state fci vec
    if args.e > 1:
        gsvec = myci.to_fci(myci.ci[0],norbci,nelec)
    else:
        gsvec = myci.to_fci(myci.ci,norbci,nelec)
    for a in range(args.e):
        # convert CISD to FCI vector
        if args.e > 1:
            esvec = myci.to_fci(myci.ci[a],norbci,nelec)
            rdm1 = myci.make_rdm1(myci.ci[a])
        else:
            esvec = myci.to_fci(myci.ci,norbci,nelec)
            rdm1 = myci.make_rdm1(myci.ci)
        if closed_shell(args.n):
            trdm = fci.direct_spin0.trans_rdm1(gsvec,esvec,norb,nelec)
        else:
            trdm = fci.direct_uhf.trans_rdm1(gsvec,esvec,norb,nelec)
        # fci.spin_op.spin_square0(esvec,norb,nelec)
        CISD_C.append({'rdm1':rdm1,'trdm':trdm,'smult':'blank'})
    # E = E_HF + E_corr
    specout['E'].update({'CISD':myci.e_corr})
    specout['C'].update({'CISD':CISD_C})
    specout['conv'].update({'CISD':myci.converged})
    sik,eik = spec_rdm(args.r0,args.l,[x['trdm'] for x in CISD_C],mf.e_tot + myci.e_corr,mf)
    specout['S'].update({'CISD':{'sik':sik,'eik':eik}})
    
    # cisd_slow
    # from pyscf.ci import cisd_slow
    # myci = cisd_slow.CISD(mf)
    # myci.verbose = 1
    # e,c = myci.kernel()
    # cisd_slow.to_fci is less featured and buggy
    # fci.spin_op.spin_square0(ci.cisd.to_fci(c[0],norb,nelec),norb,nelec)
    # list(map(lambda x: fci.spin_op.spin_square0(ci.cisd.to_fci(x,norb,nelec),norb,nelec),c))
    # list(map(lambda x,y: np.isclose(x,y).all(),c,myci.ci))
    
    return specout

def do_fci(mol,mf,args,specout):
    from pyscf import fci,ao2mo,lib
    norb = args.norb
    nelec = mol.nelec
    # mol.symmetry >>> 0
    # mol.spin >>> 0 (only for RHF)
    # direct_spin0.FCISolver(mol) (based on flow chart)
    if closed_shell(args.n):
        myfci = fci.addons.fix_spin_(fci.FCI(mf, mf.mo_coeff), .5)
    else:
        # adapted from direct_uhf.py's example
        from functools import reduce
        myfci = fci.direct_uhf.FCISolver(mf)
        nea,neb = mol.nelec
        mo_a = mf.mo_coeff[0]
        mo_b = mf.mo_coeff[1]
        h1e_a = reduce(np.dot, (mo_a.T, mf.get_hcore(), mo_a))
        h1e_b = reduce(np.dot, (mo_b.T, mf.get_hcore(), mo_b))
        g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
        g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
        g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
        g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
        g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
        g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
        h1e = (h1e_a, h1e_b)
        eri = (g2e_aa, g2e_ab, g2e_bb)
        #na = fci.cistring.num_strings(norb, nea)
        #nb = fci.cistring.num_strings(norb, neb)
    myfci.nroots = args.e
    myfci.conv_tol = 1e-7
    myfci.max_cycle = 1000
    myfci.threads = lib.num_threads()
    if closed_shell(args.n):
        myfci.kernel()
    else:
        myfci.eci,myfci.ci = myfci.kernel(h1e, eri, norb, nelec)
    FCI_C = list()
    for a in range(args.e):
        if args.e > 1:
            trdm = myfci.trans_rdm1(myfci.ci[0],myfci.ci[a],norb,nelec)
            rdm1 = myfci.make_rdm1(myfci.ci[a],norb,nelec)
            smult = fci.spin_op.spin_square0(myfci.ci[a],norb,nelec)
        else:
            trdm = 0
            rdm1 = myfci.make_rdm1(myfci.ci,norb,nelec)
            smult = fci.spin_op.spin_square0(myfci.ci,norb,nelec)
        FCI_C.append({'rdm1':rdm1,'trdm':trdm,'smult':smult})
    # myci.eci is E_HF + E_corr, not just E_corr
    specout['E'].update({'FCI':myfci.eci})
    specout['C'].update({'FCI':FCI_C})
    specout['conv'].update({'FCI':myfci.converged})
    sik,eik = spec_rdm(args.r0,args.l,[x['trdm'] for x in FCI_C],myfci.eci,mf)
    specout['S'].update({'FCI':{'sik':sik,'eik':eik}})
    
    return specout,myfci

def do_ccsd(mf,args,specout):
    '''
    CCSD + EOM-EE-CCSD + GF 
    Returns: spectra, RDM / excited state RDM, correlation energies
    '''
    from pyscf import cc
    import gf
    if closed_shell(args.n):
        mycc = cc.RCCSD(mf)
    else:
        mycc = cc.UCCSD(mf)
    Ha = constants.physical_constants['Hartree energy in eV'][0]
    ws = args.ccsd_range
    # convert eV -> Ha
    ws[0:2] /= Ha
        
    # Do the ground state CCSD calculation
    mycc.conv_tol = 1e-7 # default is 1e-7
    mycc.kernel()
    mycc.solve_lambda()
    nocc, nvir = mycc.t1.shape
    ntot = nocc+nvir
    
    CCSD_C = list()
    # Calculate the diagonal of the RDM for ground state      
    rdm = np.zeros(ntot)
    for i in range(ntot):
        rdm[i] = gf.rdm(mycc,i,i)
    CCSD_C.append({'rdm':rdm})
    	
    # Calculate and biorthonormalise the right and left eigenvectors
    e_ee, r_ee = mycc.eomee_ccsd_singlet(nroots=args.e)
    e_ee_l, l_ee = mycc.eomee_ccsd_singlet(nroots=args.e, left=True)
    if args.e != 1:
        r_ee, l_ee = mycc.biorthonormalize(r_ee,l_ee,e_ee,e_ee_l)
        # Calculate the diagonal of the RDM for each excited state   
        tot = np.zeros((ntot,args.e))
        for i in range(ntot):
            tot[i,:] = gf.e_rdm(mycc,i,i,r_ee,l_ee)
        CCSD_C.append({'erdm':tot})
    
    specout['E'].update({'CCSD':mycc.e_corr,'EOM_R':e_ee,'EOM_L':e_ee_l})
    specout['C'].update({'CCSD':CCSD_C})
    specout['conv'].update({'CCSD':mycc.converged})
    
    #----------SPECTRUM DETAILS --------------
    gf1 = gf.OneParticleGF()
    omegas = np.linspace(ws[0],ws[1],num=ws[2].astype(np.int)) # change steps to int
    # broadening, power law fit - takes r (nm) -> eV
    eta = broaden(args.r)/Ha
    gf1.gmres_tol = 1e-5
    
    #----------EE SPECTRUM SECTION --------------
    if hasattr(mf,'dmo'):
        dmo = mf.dmo
    else:
        dmo = dip_mo(args.r0,args.l,mf.mo_coeff)[2] # 3xNxN 
    #dpq = dpq.reshape(ntot,ntot,3) # transpose
    dmo = np.transpose(dmo,(1,2,0)) # NxNx3
    EEgf = gf1.solve_2pgf(mycc,range(ntot),range(ntot),range(ntot),range(ntot),omegas,eta,dmo)
    spectrum = np.zeros(omegas.size)
    for p in range(ntot):
        for r in range(ntot):
            if dmo[p,r,:].any():
                spectrum -= EEgf[p,:,r,:,:].imag.sum(axis=(0,1))
    #for p in range(ntot):
    #    for r in range(ntot):
    #        if (all(dmo[p,r,:] == 0)): continue
    #        for q in range(ntot):
    #            for s in range(ntot):
    #                spectrum -= np.imag(EEgf[p,q,r,s,:])
    spectrum /= np.pi
    specout['S'].update({'CCSD':{'sik':spectrum,'eik':omegas}})
    
    return specout

if __name__ == '__main__':
    # start the calculation
    mol,mf,args = init_pis()
    # output dictionary
    specout = {'E':dict(),'C':dict(),'S':dict(),'conv':dict(),'args':args}
    
    # AO
    if 'O' in args.T:
        specout = do_ao(args,specout)
    
    # HF
    specout,mf = do_hf(mf,args,specout)

    # All flavours of singles spectroscopy
    specout = do_singles(mf,args,specout)
    
    # CISD
    # e_corr is lowest eigenvalue, ci is lowest ev (from davidson diag)
    if 'D' in args.T:
        specout = do_cisd(mol,mf,args,specout)

    # FCI
    if 'F' in args.T:
        specout = do_fci(mol,mf,args,specout)[0]
         
    # (R/U)CCSD + EOM-EE + GF spectrum
    if 'E' in args.T:
        specout = do_ccsd(mf,args,specout)
    
    # save results ------------------------------------------------------------
    if args.j != 0:
        output_path = os.path.normpath('../output_SCF/' + args.j)
        sio.savemat(output_path,specout,oned_as='column')