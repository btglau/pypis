# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:13:24 2017

@author: Bryan Lau
"""

import numpy as np
import scipy.io as sio
import scipy.constants as sc
import os
import argparse

'''
Test change of basis code
'''

def getArgs():
    parser = argparse.ArgumentParser(description="Do a pyscf calculation on a basis set")
    parser.add_argument("fname", help="basis set file name (../basissets/[name])")
    parser.add_argument('-n',type=int,help="number of electrons",default=1)
    parser.add_argument('-e',type=float,help="dielectric",default=1)
    parser.add_argument('-r',type=float,help="radius (nm)",default=1)
    parser.add_argument('-m',type=float,help="effective mass (a.u.)",default=1)
    parser.add_argument('-j',help="file name to save results",default="blank")
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
    
    r = args.r*1E-9/sc.physical_constants['Bohr radius'][0]
    r = args.r # atomic units input
    E_scale = 1/(2*args.m*r**2)
    
    # midway defaults to 2.7, but I build pyscf with the anaconda3 module
    print('PIS pyscf')
    print('Basis set: {0}'.format(args.fname))
    print('{0} electrons'.format(args.n))
    
    # load the mat file
    basisset_path = os.path.normpath('../basissets/' + args.fname)
    mat_contents = sio.loadmat(basisset_path,matlab_compatible=True)
    
    U = mat_contents['Ur'] + 1j*mat_contents['Ui']
    eri_chem = mat_contents['eri'].copy()
    # go to physics notation
    eri_phys = eri_chem.copy().transpose(0,2,1,3)
    
    print("Pre-transform check for 4-fold symmetry (phys):")
    print("<01|23> =? <10|32> :", np.allclose(eri_phys,eri_phys.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (phys):")
    print("<01|23> =? <03|21> :", np.allclose(eri_phys,eri_phys.transpose(0,3,2,1)))
    
    eri_chem = eri_phys.copy().transpose(0,2,1,3)
    print("Pre-transform check for 4-fold symmetry (chem):")
    print("<01|23> =? <10|32> :", np.allclose(eri_chem,eri_chem.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (chem):")
    print("<01|23> =? <01|32> :", np.allclose(eri_chem,eri_chem.transpose(0,1,3,2)))
    
    # transform
    print('')
    print('Transforming from complex to real orbitals ...')
    eri_phys_real = change_basis_2el_complex(eri_phys,U)
    print('Max/min imag', eri_phys_real.imag.max(), ' ', eri_phys_real.imag.min())
    eri_phys_real = eri_phys_real.real
    print('')
    
    print("Checking 4-fold symmetry (phys):")
    print("<01|23> =? <10|32> :", np.allclose(eri_phys_real,eri_phys_real.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (phys):")
    print("<01|23> =? <03|21> :", np.allclose(eri_phys_real,eri_phys_real.transpose(0,3,2,1)))
    
    eri_chem_real = eri_phys_real.copy().transpose(0,2,1,3)
    print("Checking 4-fold symmetry (chem):")
    print("<01|23> =? <10|32> :", np.allclose(eri_chem_real,eri_chem_real.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (chem):")
    print("<01|23> =? <01|32> :", np.allclose(eri_chem_real,eri_chem_real.transpose(0,1,3,2)))
    
    # try another way
    print('')
    print('pure way without intermediate steps')
    eri_phys_real = change_basis_2el_complex(mat_contents['eri'].transpose(0,2,1,3),U).real
    eri_chem_real = eri_phys_real.copy().transpose(0,2,1,3)  
    
    print("Checking 4-fold symmetry (phys):")
    print("<01|23> =? <10|32> :", np.allclose(eri_phys_real,eri_phys_real.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (phys):")
    print("<01|23> =? <03|21> :", np.allclose(eri_phys_real,eri_phys_real.transpose(0,3,2,1)))
    
    eri_chem_real = eri_phys_real.copy().transpose(0,2,1,3)
    print("Checking 4-fold symmetry (chem):")
    print("<01|23> =? <10|32> :", np.allclose(eri_chem_real,eri_chem_real.transpose(1,0,3,2)))
    print("Checking 8-fold symmetry (chem):")
    print("<01|23> =? <01|32> :", np.allclose(eri_chem_real,eri_chem_real.transpose(0,1,3,2)))