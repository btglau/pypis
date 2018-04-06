# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:27:54 2018

directly diagonalize A/B methods
# python -u -i pis_ab_direct.py -l 8 -e 1 -n 2 -r 1 -s 0.25

@author: Bryan Lau
"""

import numpy as np
import time

from pyscf import tddft
from pyscf.tddft import rhf_slow
from scipy import linalg

def pickeig(w, v):
    # We only need positive eigenvalues
    realidx = np.where((abs(w.imag) < 1e-6) & (w.real > 0))[0]
    idx = realidx[w[realidx].real.argsort()]
    return w[idx].real, v[:,idx].real, idx

def norm_xy(td,z):
    from pyscf import lib
    nocc = (td._scf.mo_occ>0).sum()
    nmo = td._scf.mo_occ.size
    nvir = nmo - nocc
    # normalize XY eigenvector from A/B methods
    x, y = z.reshape(2,nvir,nocc)
    norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
    norm = 1/np.sqrt(norm)
    return x*norm, y*norm

def direct_TDHF(mf):
    '''
    A/B w/ exchange (=TDHF/RPA)
    '''
    print()
    print('************************************************')
    print('Diagonalization of full A/B hamiltonian for TDHF')
    print('************************************************')
    td = tddft.TDHF(mf)
    vind,hdiag = td.gen_vind(td._scf)
    print('Building matrix ...')
    start_time = time.time()
    H = vind(np.identity(hdiag.size))
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    w,v = linalg.eig(H)
    w,v,idx = pickeig(w,v)
    # list comprehensions grab by row, so transpose V
    xy = [norm_xy(td,z) for z in v.T]
    td.e = w
    td.xy = xy
    return td

def direct_dRPA(mf):
    '''
    A/B w/o exchange (direct RPA), i.e. TDH
    
    Returns a td object (dRPA/TDH -> TDHF -> rhf.TDHF)
    '''
    print()
    print('************************************************************')
    print('Diagonalization of full A/B hamiltonian for direct RPA / TDH')
    print('************************************************************')
    td = rhf_slow.dRPA(mf)
    td.eris = td.ao2mo()
    vind,hdiag = td.gen_vind(mf)
    print('Building matrix ...')
    start_time = time.time()
    H = vind(np.identity(hdiag.size))
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    w,v = linalg.eig(H)
    w,v,idx = pickeig(w,v)
    xy = [norm_xy(td,z) for z in v.T]
    td.e = w
    td.xy = xy
    return td

if __name__ == '__main__':
    import pis_hf
    # start the calculation
    mol,mf,args = pis_hf.init_pis()
    # output dictionary
    specout = {'E':dict(),'C':dict(),'S':dict(),'conv':dict()}
    
    # HF
    mf,specout = pis_hf.do_hf(mf,args,specout)
    
    # TDHF
    my_TDHF = direct_TDHF(mf)
    
    # dRPA
    my_dRPA = direct_dRPA(mf)