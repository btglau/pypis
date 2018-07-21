#!/usr/bin/env python
#
# Author: Bryan Lau
# -*- coding: utf-8 -*-

'''
A modification of pyscf 1.5 to allow custom integrals to be used. Also, left
eigenvectors for EE EOM CCSD (neutral excitations)
'''

import numpy as np
from pyscf import lib

def leomee_ccsd_matvec_singlet(self, vector):
    if not hasattr(self,'imds'):
        self.imds = _IMDS(self)
    if not self.imds.made_ee_imds:
        self.imds.make_ee()
    imds = self.imds

    r1, r2 = self.vector_to_amplitudes(vector)
    t1, t2, eris = self.t1, self.t2, self.eris
    nocc, nvir = t1.shape

    #rho = r2*2 - r2.transpose(0,1,3,2)
    Hr1  = lib.einsum('ae,ia->ie', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ia->ma', imds.Foo, r1)
    Hr2 = 2*lib.einsum('me,ia->imae',imds.Fov, r1)
    Hr2 -= lib.einsum('me,ia->imea',imds.Fov, r1)

    Hr2+= lib.einsum('mnij,ijab->mnab', imds.woOoO, r2) * .5
    Hr2+= lib.einsum('be,ijab->ijae', imds.Fvv   , r2)
    Hr2-= lib.einsum('mj,ijab->imab', imds.Foo   , r2)
   
    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
    #:Hr1 += lib.einsum('mfae,imef->ia', eris_ovvv, rho)
    #:tmp = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2)
    #:Hr2 -= lib.einsum('ma,mbij->ijab', t1, tmp)
    #:tmp  = lib.einsum('meaf,me->af', eris_ovvv, r1) * 2
    #:tmp -= lib.einsum('mfae,me->af', eris_ovvv, r1)
    mem_now = lib.current_memory()[0]
    max_memory = lib.param.MAX_MEMORY - mem_now
    blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
    tau2 = make_tau(r2, r1, t1, fac=2)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
        ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
        Hr2 += 2*lib.einsum('mfae,ia->imef', ovvv, r1[p0:p1])
        Hr2 -= lib.einsum('mfae,ia->imfe', ovvv, r1[p0:p1])
        tmp = lib.einsum('ma,ijab->mbij', t1[p0:p1], tau2)
        Hr2 -= lib.einsum('meaf,maij->ijef', ovvv, tmp)
       
        tmp = lib.einsum('ijab,ijfb->af', r2[:,p0:p1], t2)
        Hr1 += lib.einsum('meaf,af->me', ovvv, tmp) * 2
        Hr1 -= lib.einsum('mfae,af->me', ovvv, tmp)
       
        ovvv = tmp = None

    Hr1 -= lib.einsum('mbij,ijab->ma', imds.woVoO, r2)

    Hr2-= 2*lib.einsum('mnie,ia->mnae', imds.woOoV, r1)
    Hr2+= lib.einsum('mnie,ia->mnea', imds.woOoV, r1)

    tmp = lib.einsum('ijab,njab->ni', r2, t2)
    Hr1-= lib.einsum('nmie,ni->me', imds.woOoV, tmp) * 2
    Hr1+= lib.einsum('mnie,ni->me', imds.woOoV, tmp)
    tmp = None

    for p0, p1 in lib.prange(0, nvir, nocc):
        Hr1 += lib.einsum('ejab,ijab->ie', np.asarray(imds.wvOvV[:,:,p0:p1]), r2[:,:,p0:p1])

    oVVo = np.asarray(imds.woVVo)
    Hr2 += lib.einsum('mbej,jiab->imea', oVVo, r2)
    Hr2 += lib.einsum('mbej,jiba->imea', oVVo, r2) * .5
    oVvO = np.asarray(imds.woVvO) + oVVo * .5
    oVVo = tmp = None
    Hr1 += lib.einsum('maei,ia->me', oVvO, r1) * 2
    Hr2 += 2*lib.einsum('mbej,ijab->imae', oVvO, r2)
    Hr2 -= lib.einsum('mbej,ijab->imea', oVvO, r2)
    oVvO = None

    eris_ovov = np.asarray(eris.ovov)
    tau2 = make_tau(r2, r1, t1, fac=2)
    tau = make_tau(t2, t1, t1)
    tmp = lib.einsum('ijab,mnab->mnij', tau2, tau) * .5
    tau2 = None
    Hr2 += lib.einsum('menf,mnij->ijef', eris_ovov, tmp)
    tau = tmp = None

    tmp = lib.einsum('na,ia->ni', t1, r1)
    Hr2 -= 2*lib.einsum('nemf,ni->imef', eris_ovov, tmp)
    Hr2 += lib.einsum('nemf,ni->imfe', eris_ovov, tmp)
   
    tmp = lib.einsum('ijba,miab->mj', r2, t2)
    Hr2 -= 2*lib.einsum('nemf,ni->imef', eris_ovov, tmp)
    Hr2 += lib.einsum('nemf,ni->imfe', eris_ovov, tmp)
    tmp = None

    tmp  = lib.einsum('jiab,ijea->eb', r2, t2)
    tmp  = lib.einsum('eb,nb->en', tmp, t1)
    Hr1 -= lib.einsum('mfne,en->mf', eris_ovov, tmp) * 2
    Hr1 += lib.einsum('menf,en->mf', eris_ovov, tmp)
   
    tmp = lib.einsum('jiab,ijea->eb', r2, t2)
    Hr2 -= 2*lib.einsum('menf,eb->mnbf', eris_ovov, tmp)
    Hr2 += lib.einsum('menf,eb->mnfb', eris_ovov, tmp)
    tmp = eris_ovov = rho = None

    tau2 = make_tau(r2, r1, t1, fac=2) # This may need editing.
    eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
    Hr2 += lib.einsum('ijab,aebf->ijef', tau2, eris_vvvv) * .5
    #l_add_vvvv_(self, tau2, eris, Hr2)
    tau2 = None

    Hr2 = Hr2 + Hr2.transpose(1,0,3,2)
    vector = self.amplitudes_to_vector(Hr1, Hr2)
    return vector

def biorthonormalize(self,r_ee,l_ee,eee_r,eee_l):
    if (len(r_ee) != len(l_ee)):
        print('The number of right and left hand eigenvectors must be the same in order to biorthonormalise them')
        return r_ee, l_ee
    if (np.any(np.round(eee_r,4) != np.round(eee_l,4))):
        print('The right and left hand eigenvalues must agree to at least 4 decimal places in order to biorthonormalise the eigenvectors')
        return r_ee, l_ee
    deg_range = 0
    deg_start = 0
    deg = False
    EEnroots = len(r_ee)
    for i in range(EEnroots):
        if (i < EEnroots-1) and round(eee_r[i],3) == round(eee_r[i+1],3):
            deg_range += 1
            if not deg:
                deg = True
                deg_start = i
        else:
            if deg_range == 0:
                pass
            else:
                deg_range += 1
                lvecs = np.asarray(l_ee[deg_start:deg_start+deg_range])
                rvecs = np.asarray(r_ee[deg_start:deg_start+deg_range])
                lvecs = np.linalg.qr(lvecs.transpose(1,0))[0].transpose(1,0)
                rvecs = np.linalg.qr(rvecs.transpose(1,0))[0].transpose(1,0)
                
                for j in range(deg_range):
                    k = j + deg_start
                    l_ee[k] = lvecs[j,:]
                    r_ee[k] = rvecs[j,:]
                deg = False
                deg_range = 0
                
    lvecs = np.asarray(l_ee)
    rvecs = np.asarray(r_ee)
    M = np.inner(lvecs,rvecs)
    P,L,U = scipy.linalg.lu(M)
    lvecs = np.matmul(P.transpose(1,0),lvecs)
    Linv = np.linalg.inv(L)
    Uinv = np.linalg.inv(U)
    lvecs = np.matmul(Linv, lvecs)
    rvecs = np.matmul(rvecs.transpose(1,0),Uinv).transpose(1,0)
    for i in range(EEnroots):
        l_ee[i] = lvecs[i,:]
        r_ee[i] = rvecs[i,:]
                
    r_ee = self.normalize_r_ee(r_ee)
    l_ee = self.normalize_l_ee(r_ee,l_ee)
    
    #check = np.inner(r_ee,l_ee)
    #mask = np.ones(check.shape, dtype=bool)
    #np.fill_diagonal(mask, 0)
    #max_value = np.abs(check[mask]).max()
    #print max_value,np.trace(check)
    
    return r_ee,l_ee

def get_Hrow(self):
    Hr1 = imd.cc_Fov(self.t1,self.t2,self.eris)
    Hr2 = np.copy(self.eris.ovov)
    self.Hrow = self.amplitudes_to_vector(Hr1,Hr2)
    
    return self.amplitudes_to_vector(Hr1,Hr2)

def normalize_l_ee(self,r_ee,l_ee):
    EEnroots = len(r_ee)
    for i in range(EEnroots):
        n = np.dot(l_ee[i],r_ee[i])
        l_ee[i] /= n
    return l_ee

def normalize_r_ee(self,r_ee):
    r0 = self.get_r0(r_ee)
    EEnroots = len(r_ee)
    for i in range(EEnroots):
        n = np.dot(r_ee[i],r_ee[i]) + r0[i]*r0[i]
        n = np.sqrt(n)
        r_ee[i] /= n
        r0[i] /= n
    return r_ee