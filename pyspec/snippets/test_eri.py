# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:18:43 2018

@author: Bryan Lau

eri test

can an existing 8 fold eri be reindexed to pyscf's 8 fold eri?
won't work, because the i j k l indexing goes out to (n,n,n,n)
"""

import pis_hf,numpy
from pyscf import ao2mo

mol,mf,args = pis_hf.init_pis()
eri = mf._eri
norb = args.norb
npair = norb*(norb+1)//2
eri1 = numpy.empty(npair*(npair+1)//2, dtype=eri.dtype)
# _call_restore(origsym, targetsym, eri, eri1, norb, tao=None)
eri1 = ao2mo.addons._call_restore(1,8,eri,eri1,norb)