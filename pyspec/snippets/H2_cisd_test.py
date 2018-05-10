# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:29:32 2018

@author: Bryan Lau
"""

from pyscf import gto, scf, ci, cc, fci
from pyscf.ci import cisd_slow
import numpy as np

mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)
mf.kernel()
norb = mf.mo_coeff.shape[1]
nelec = mol.nelec

myci = ci.CISD(mf)
myci.nroots = 10
e0,c0 = myci.kernel()

mycc = cc.RCCSD(mf)
mycc.kernel()
e1,c1 = mycc.eomee_ccsd_singlet(nroots=10)

mycislow = ci.cisd_slow.CISD(mf)
mycislow.verbose = 1
e2,c2 = mycislow.kernel()

print(list(map(lambda x: fci.spin_op.spin_square0(ci.cisd.to_fci(x,norb,nelec),norb,nelec)[1],c0)))
print(list(map(lambda x: fci.spin_op.spin_square0(ci.cisd.to_fci(x,norb,nelec),norb,nelec)[1],c2)))

print('RCCSD, CISD, CISD slow, FCI')
print(np.concatenate((np.array([mf.e_tot+mycc.e_corr]),e1+mf.e_tot+mycc.e_corr)))
print(e0+mf.e_tot)
print(e2+mf.e_tot)

##myfci = fci.addons.fix_spin_(fci.FCI(mf, mf.mo_coeff), .5)
#myfci = fci.FCI(mf, mf.mo_coeff)
#myfci.nroots = 10
#e3,c3 = myfci.kernel()
#list(map(lambda x: fci.spin_op.spin_square0(x,norb,nelec)[1],myfci.ci))
#print(e3)