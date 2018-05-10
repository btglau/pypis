# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:54:32 2018

Transform PIS 4-fold eri's to 8-fold eri's, to save space

@author: Bryan Lau
"""

import numpy as np
from pyscf import ao2mo
import scipy.io as sio
import glob, os

def four_to_eight(g,C):
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
    print('4 fold to 8 fold transform')
    
    mypath = os.path.normpath('../basissets_4fold/' + '*.mat')
    mysets = glob.glob(mypath)
    
    for aset in mysets:
        print('Transforming ' + aset)
        mat_contents = sio.loadmat(aset,matlab_compatible=True)
        norb = int(mat_contents['args']['N'].item())
        U = mat_contents['Ur'] + 1j*mat_contents['Ui']
        pis_eri = ao2mo.restore(8,four_to_eight(mat_contents['eri'].transpose(0,2,1,3),U).real.transpose(0,2,1,3),norb)
        mat_contents['eri'] = pis_eri
        # save as simplified filename
        output_path = os.path.normpath('../basissets/' + aset.split('_')[3] + '_alt')
        print('Saving transformed to ' + output_path)
        sio.savemat(output_path,mat_contents)