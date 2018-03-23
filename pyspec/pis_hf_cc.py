# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:13:24 2017

@author: Bryan Lau

1/19/2018 - change CISD to FCI, to get singlet states easily
"""

import scipy.io as sio
import scipy.constants as sc
import numpy as np
import os
import argparse
from pyscf import gto, scf, ao2mo, ci, cc, tddft, fci
from pyscf.cc import gf

'''
upon execution, this program should look up a directory, then into a folder
called 'basissets'. it will then look at the existing mat files, and if a
calculation has not been done on them, load them and run some HF/pHF
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
    parser = argparse.ArgumentParser(description="Do a pyscf calculation on a basis set")
    parser.add_argument("fname", help="basis set file name (../basissets/[name])")
    parser.add_argument('-n',type=int,help="number of electrons",default=1)
    parser.add_argument('-d',type=float,help="dielectric constant",default=1)
    parser.add_argument('-e',type=int,help="number of excitations (1 = ground state only)",default=1)
    parser.add_argument('-r',type=float,help="radius (nm)",default=1)
    parser.add_argument('-m',type=float,help="effective mass",default=1)
    parser.add_argument('-j',help="file name to save results",default="blank")
    parser.add_argument('--methods',help="post-HF methods; give an unordered string with options needed: (C)IS, (T)DHF, CIS(D), (F)CI, (E)OM-CCSD')",default='CDE')
    # By default, the arguments are taken from sys.argv[1:]
    return parser.parse_args()

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

if __name__ == '__main__':
    # parse the input
    args = getArgs()
    
    rnm = args.r
    r = rnm*1E-9/sc.physical_constants['Bohr radius'][0] # nm input
    l = args.fname[args.fname.index("l") + 1]
    #r = args.r # atomic units input
    E_scale = 1/(2*args.m*r**2)
    
    # python 3 style printing
    print('PIS pyscf')
    print('Basis set: {0}'.format(args.fname))
    print('{0} electrons'.format(args.n))
    
    # load the mat file
    basisset_path = os.path.normpath('./basissets/' + args.fname)
    mat_contents = sio.loadmat(basisset_path,matlab_compatible=True)

    mol = gto.M()
    mol.verbose = 4
    # if nelectrons is None (initial call to function), then it sets it to
    # tot_electrons = (sum of atom charges - charge)
    # nelectrons = total electrons
    mol.nelectron = args.n
    # prevent pHF from crashing, i.e. force it to use provided AOs
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    # set the core, overlap, and eri integrals
    mf.get_hcore = lambda *args: mat_contents['Hcore'] * E_scale
    mf.get_ovlp = lambda *args: mat_contents['ovlp']
    norbs = int(mat_contents['args']['N'].item())
    mf.init_guess = '1e'
    
    # because scipy.io can't load complex matrices
    U = mat_contents['Ur'] + 1j*mat_contents['Ui']
    # go to physics notation
    eri_phys_real = change_basis_2el_complex(mat_contents['eri'].transpose(0,2,1,3),U).real    
    # back to chemists notation
    mf._eri = ao2mo.restore(8,eri_phys_real.transpose(0,2,1,3) / args.d / r,norbs)
    
    #----------CCSD SECTION --------------

    mf.run()
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-6
    e_cc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    nocc, nvir = t1.shape
    
    #----------SPECTRUM DETAILS --------------
    
    gf1 = gf.OneParticleGF()
    dw = 0.001
    wmin = 0.0
    wmax = 0.01
    nw = int((wmax-wmin)/dw) + 1
    omegas = np.linspace(wmin, wmax, nw)
    eta = 0.000125
    gf1.gmres_tol = 1e-5
    
    ntot = nocc + nvir
    
    dpq = np.loadtxt("dpq_l4_r1_d10_m25.txt",complex)    
    dpq = dpq.reshape(ntot,ntot,3)
    
    #----------EE SPECTRUM SECTION --------------

    EEgf = gf1.solve_2pgf(mycc,range(ntot),range(ntot),range(ntot),range(ntot),omegas,eta,dpq)

    spectrum = np.zeros((len(omegas)))
    for p in range(ntot):
        for r in range(ntot):
            if (all(dpq[p,r,:] == 0)): continue
            for q in range(ntot):
                for s in range(ntot):
                    spectrum -= np.imag(EEgf[p,q,r,s,:])
    spectrum /= np.pi
        
    with open("EEspectrum_l"+str(l)+"_r"+str(int(rnm))+".txt", "w") as f:
        for i,s in enumerate(spectrum):
            f.write(str(wmin+dw*i) + "    " + str(s) + "\n")
