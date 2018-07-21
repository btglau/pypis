# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:27:54 2018

directly diagonalize A/B methods
# python -u -i pis_ab_direct.py -l 8 -e 1 -n 2 -r 1 -s 0.25

@author: Bryan Lau
"""

import numpy as np
import time
import rhf_slow
from scipy import linalg

from pyscf import lib
from pyscf.lib import logger

def pickeig(w, v):
    # We only need positive eigenvalues
    realidx = np.where((abs(w.imag) < 1e-6) & (w.real > 0))[0]
    idx = realidx[w[realidx].real.argsort()]
    return w[idx].real, v[:,idx].real, idx

def norm_xy(td, z):
    nocc = td.nocc
    nvir = td.nmo - nocc
    x, y = z.reshape(2,nvir,nocc)
    norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
    norm = 1/np.sqrt(norm)
    return x*norm, y*norm

def direct(td):
    '''
    Direct diagonalization, calling methods in the td object
    '''
    log = logger.Logger(td._scf.stdout, td._scf.verbose)
    log.info('\n')
    log.info('******** {} for {} ********'.format(
             td.__class__, td._scf.__class__))
    log.info('    Diagonalize the full A/B singles Hamiltonian')
    td.eris = td.ao2mo()
    vind,hdiag = td.gen_vind(td._scf)
    cput0 = time.clock(), time.time()
    H = vind(np.identity(hdiag.size))
    w,v = linalg.eig(H)
    log.timer('build & diagonalize H',*cput0)
    w,v,idx = pickeig(w,v)
    # list comprehensions grab by row, so transpose V
    td.xy = [norm_xy(td,z) for z in v.T]
    td.e = w
    return td

if __name__ == '__main__':
    import pis_hf
    # start the calculation
    mol,mf,args = pis_hf.init_pis()
    # output dictionary
    specout = {'E':dict(),'C':dict(),'S':dict(),'conv':dict()}
    
    # HF
    specout,mf = pis_hf.do_hf(mf,args,specout)
    
    # TDHF
    td = rhf_slow.TDHF(mf)
    td = direct(td)
    
    td2 = rhf_slow.TDHF(mf)
    td2.direct()
    
    print('Energy same? {}'.format(np.allclose(td.e,td2.e)))
    ev = 0
    for ev1,ev2 in zip(td.xy,td2.xy):
        print('Eigenvalue {}: {} {}'.format(ev,np.allclose(ev1[0],ev2[0]),np.allclose(ev1[1],ev2[1])))
        ev += 1
    
    sik,eik = pis_hf.spec_singles(args.r0,args.l,mf,td)
    sik2,eik2 = pis_hf.spec_singles(args.r0,args.l,mf,td2)
    
    print('Check sik,eik: {} {}'.format(np.allclose(sik,sik2),np.allclose(eik,eik2)))
    
#    # dRPA
#    td = rhf_slow.dRPA(mf)
#    td.eris = td.ao2mo()
#    td = direct_dRPA(td)
#    dxy = copy.copy(td.xy)
#    de = copy.copy(td.e)
#    td.direct()
#       
#    print('Energy same? {}'.format(np.allclose(td.e,de)))
#    ev = 0
#    for x,y in zip(td.xy,dxy):
#        print('Eigenvalue {}: {} {}'.format(ev,np.allclose(x[0].T,y[0]),np.allclose(x[1].T,y[1])))
#        ev += 1