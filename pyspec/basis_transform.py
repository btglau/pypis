# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:54:32 2018

Transform PIS 4-fold eri's to 8-fold eri's, to save space

@author: Bryan Lau
"""

from pis_hf import change_basis_2el_complex as four_to_eight
from pyscf import ao2mo
import scipy.io as sio
import glob, os

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
        output_path = os.path.normpath('../basissets/' + aset.split('_')[3])
        print('Saving transformed to ' + output_path)
        sio.savemat(output_path,mat_contents)