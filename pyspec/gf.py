import collections
import sys

import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy
from pyscf import lib

def greens_func_multiply(ham,vector,linear_part,args=None):
    return np.array(ham(vector) + (linear_part)*vector)
    
##################################
# Reduced Density Matrices       #
##################################

def rdm(cc,p,q):
    return greens_e0_ee_rhf(cc,p,q) + 2*(q == s)
   
def tdm(cc,p,q,r_ee):
    if cc.r0 is None: cc.r0 = cc.get_r0(r_ee)
    nstates = len(r_ee)
    tdm_el = np.zeros(nstates, dtype = r_ee.dtype)
    e_vector = greens_e_vector_ee_rhf(cc,p,q).astype(r_ee.dtype)
    e0 = rdm(cc,p,q)
    r0 = cc.r0
    tdm_el += np.einsum('i,vi->v', e_vector,r_ee)
    tdm_el += e0*r0
    
    return tdm_el
    
def e_rdm(cc,p,q,r_ee,l_ee):
    nocc,nvir = cc.t1.shape
    nstates = len(r_ee)
    el = np.zeros(nstates)
    r0 = cc.get_r0(r_ee)
    for i in range(nstates):
        l1_ee , l2_ee = cc.vector_to_amplitudes(l_ee[i])
        e_vector = np.real(greens_e_vector_ta_rhf(cc,p,q,l1_ee,l2_ee))
        e0 = greens_e0_ta_rhf(cc,p,q,l1_ee,l2_ee)
        if (p < nocc): el[i] = 2*(p == q)
        if (p < nocc) and (q >= nocc): el[i] = 2*cc.t1[p,q]
        el[i] += np.dot(e_vector,r_ee[i])
        el[i] += e0*r0[i]
        
    return el
    
###################
#TA Greens       #
###################
        
def greens_e_vector_ta_rhf(cc,p,q,l1_ee,l2_ee):
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=np.complex)
    if p < nocc:
        if q < nocc:
            vector1[p,:] -= l1_ee[q,:]
            
            vector1 += np.einsum('iab,a->ib', l2_ee[:,q,:,:], cc.t1[p,:]) #13
            vector1 -= 2*np.einsum('iab,a->ib', l2_ee[q,:,:,:], cc.t1[p,:]) #13
            
            vector2[:,p,:,:] += l2_ee[q,:,:,:]
            vector2[p,:,:,:] += l2_ee[q,:,:,:].transpose(0,2,1) #12
            vector2[:,p,:,:] -= 2*l2_ee[q,:,:,:].transpose(0,2,1)
            vector2[p,:,:,:] -= 2*l2_ee[q,:,:,:] #12
            
            for j in range(nvir):
                vector2[q,q,j,j] *= 0.5
                
        else:
            qp = q - nocc
            
            vector1[p,:] -= np.einsum('ia,i->a', l1_ee, cc.t1[:,qp])
            vector1[:,qp] -= np.einsum('ia,a->i', l1_ee, cc.t1[p,:])
            
            vector1 += 2*np.einsum('ijab,jb->ia', l2_ee, cc.t2[p,:,qp,:]) #14
            vector1 -= np.einsum('ijab,jb->ia', l2_ee, cc.t2[p,:,:,qp])
            
            vector1[p,:] -= 2*np.einsum('ijab,ijb->a', l2_ee, cc.t2[:,:,qp,:])
            vector1[p,:] += np.einsum('ijab,ijb->a', l2_ee, cc.t2[:,:,:,qp]) #15
            
            vector1[:,qp] += np.einsum('ijab,jab->i', l2_ee, cc.t2[:,p,:,:]) #16
            vector1[:,qp] -= 2*np.einsum('ijab,jab->i', l2_ee, cc.t2[p,:,:,:])
            
            vector1 -= np.einsum('ijab,a,j->ib', l2_ee, cc.t1[p,:],cc.t1[:,qp])
            vector1 += 2*np.einsum('ijab,b,j->ia', l2_ee, cc.t1[p,:],cc.t1[:,qp])
           
            vector2[p,:,qp,:] += 2*l1_ee
            vector2[:,p,qp,:] -= l1_ee
            vector2[:,p,:,qp] += 2*l1_ee
            vector2[p,:,:,qp] -= l1_ee
            
            vector2[:,p,:,:] -= 2*np.einsum('ijab,j->iab', l2_ee, cc.t1[:,qp])
            vector2[p,:,:,:] += np.einsum('ijab,j->iab', l2_ee, cc.t1[:,qp])
            vector2[p,:,:,:] -= 2*np.einsum('ijab,j->iba', l2_ee, cc.t1[:,qp])
            vector2[:,p,:,:] += np.einsum('ijab,j->iba', l2_ee, cc.t1[:,qp])
            
            vector2[:,:,qp,:] += 2*np.einsum('ijab,b->jia', l2_ee, cc.t1[p,:]) #19
            vector2[:,:,:,qp] -= np.einsum('ijab,b->jia', l2_ee, cc.t1[p,:])
            vector2[:,:,:,qp] += 2*np.einsum('ijab,b->ija', l2_ee, cc.t1[p,:]) #19
            vector2[:,:,qp,:] -= np.einsum('ijab,b->ija', l2_ee, cc.t1[p,:])
            
            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5
                    
    else:
        pp = p - nocc
        if q < nocc:
            vector1 += 2*l2_ee[q,:,pp,:] #9
            vector1 -= l2_ee[q,:,:,pp]
            
        else:
            qp = q - nocc
            vector1[:,qp] += l1_ee[:,pp] #3
            vector1 += 2*np.einsum('ija,j->ia', l2_ee[:,:,:,pp], cc.t1[:,qp]) #11
            vector1 -= np.einsum('ija,j->ia', l2_ee[:,:,pp,:], cc.t1[:,qp])

            vector2[:,:,qp,:] -= l2_ee[:,:,pp,:].transpose(1,0,2)
            vector2[:,:,:,qp] -= l2_ee[:,:,pp,:]    
            vector2[:,:,qp,:] += 2*l2_ee[:,:,pp,:]  #10
            vector2[:,:,:,qp] += 2*l2_ee[:,:,pp,:].transpose(1,0,2)  #10  
            
            for i in range(nocc):
                vector2[i,i,qp,qp] *= 0.5
                
    return cc.amplitudes_to_vector(vector1,vector2)
    
def greens_e0_ta_rhf(cc,p,q,l1_ee,l2_ee):
    nocc, nvir = cc.t1.shape
    if p < nocc:
        if q < nocc:
            e0 = np.dot(cc.t1[p,:],l1_ee[q,:])
            
            e0 -= 2*np.dot(cc.t2[p,:,:,:].reshape(-1),l2_ee[q,:,:,:].reshape(-1))
            e0 += np.dot(cc.t2[p,:,:,:].reshape(-1),cc.l2[:,q,:,:].reshape(-1))
            
        else:
            qp = q - nocc
            
            e0 = 2*np.dot(l1_ee.reshape(-1),cc.t2[p,:,qp,:].reshape(-1))
            e0 -= np.dot(l1_ee.reshape(-1),cc.t2[:,p,qp,:].reshape(-1))
            
            e0 -= np.dot(l1_ee.reshape(-1),np.einsum('i,a->ia',cc.t1[:,qp],cc.t1[p,:]).reshape(-1))
            
            tmp = np.einsum('ija,b->ijab',cc.t2[:,:,qp,:],cc.t1[p,:])
            tmp -= 2*np.einsum('ijb,a->ijab',cc.t2[:,:,qp,:],cc.t1[p,:])
            tmp -= np.einsum('iab,j->ijab',cc.t2[p,:,:,:],cc.t1[:,qp])
            tmp += 2*np.einsum('jab,i->ijab',cc.t2[p,:,:,:],cc.t1[:,qp])
            e0 -= np.dot(l2_ee.reshape(-1),tmp.reshape(-1))
    else:
        pp = p - nocc
        if q < nocc:
            e0 = l1_ee[q,pp]
        else:
            qp = q - nocc
            
            e0 = np.dot(cc.t1[:,qp],l1_ee[:,pp])
            
            e0 += 2*np.dot(cc.t2[:,:,qp,:].reshape(-1),l2_ee[:,:,pp,:].reshape(-1))
            e0 -= np.dot(cc.t2[:,:,qp,:].reshape(-1),l2_ee[:,:,:,pp].reshape(-1))
    
    return e0

            
            

###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to minus
        vector1 += -cc.t1[p,:]
        vector2 += -cc.t2[p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return cc.amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to plus
        vector1 += cc.l1[p,:]
        vector2 += (2*cc.l2[p,:,:,:] - cc.l2[:,p,:,:])
        pass
    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', cc.l1, cc.t1[:,p-nocc])
        vector1 += 2*np.einsum('klca,klc->a', cc.l2, cc.t2[:,:,:,p-nocc])
        vector1 -=   np.einsum('klca,lkc->a', cc.l2, cc.t2[:,:,:,p-nocc])

        vector2[:,p-nocc,:] += -2.*cc.l1
        vector2[:,:,p-nocc] += cc.l1
        vector2 += 2*np.einsum('k,jkba->jab', cc.t1[:,p-nocc], cc.l2)
        vector2 -=   np.einsum('k,jkab->jab', cc.t1[:,p-nocc], cc.l2)
    return cc.amplitudes_to_vector_ea(vector1,vector2)
    
def initial_ea_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=np.complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=np.complex)
    return cc.amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    if p < nocc:
        vector1[ p ] = 1.0
    else:
        vector1 += cc.t1[:,p-nocc]
        vector2 += cc.t2[:,:,:,p-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    if p < nocc:
        vector1[ p ] = -1.0
        vector1 += np.einsum('ia,a->i', cc.l1, cc.t1[p,:])
        vector1 += 2*np.einsum('ilcd,lcd->i', cc.l2, cc.t2[p,:,:,:])
        vector1 -=   np.einsum('ilcd,ldc->i', cc.l2, cc.t2[p,:,:,:])

        vector2[p,:,:] += -2.*cc.l1
        vector2[:,p,:] += cc.l1
        vector2 += 2*np.einsum('c,ijcb->ijb', cc.t1[p,:], cc.l2)
        vector2 -=   np.einsum('c,jicb->ijb', cc.t1[p,:], cc.l2)
    else:
        vector1 += -cc.l1[:,p-nocc]
        vector2 += -2*cc.l2[:,:,p-nocc,:] + cc.l2[:,:,:,p-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def initial_ip_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    return cc.amplitudes_to_vector_ip(vector1,vector2)
    
###################
# 2PPE Greens       #
###################

def greens_b_vector_2ppe_rhf(cc,p,r1_ee,r2_ee):
    nocc, nvir = cc.t1.shape
    r1_ee = r1_ee.reshape((nocc, nvir))
    r2_ee = r2_ee.reshape((nocc,nocc, nvir, nvir))
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    if p < nocc:
        vector2[:,p,:] -= r1_ee[:,:]
        vector2[p,:,:] += 2*r1_ee[:,:]
    else:
        vector1 += r1_ee[:,p-nocc]
        
        vector2 += 2*r2_ee[:,:,p-nocc,:]
        vector2 -= r2_ee[:,:,p-nocc,:].transpose(1,0,2)
        
        vector2 += 2*np.einsum('i,ja->ija', cc.t1[:,p-nocc],r1_ee)
        vector2 -= np.einsum('j,ia->ija', cc.t1[:,p-nocc],r1_ee)
        
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_2ppe_rhf(cc,q,l1_ee,l2_ee):
    nocc, nvir = cc.t1.shape
    l1_ee = l1_ee.reshape((nocc, nvir))
    l2_ee = l2_ee.reshape((nocc,nocc, nvir, nvir))
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    if q < nocc:
        vector1 -= np.einsum('ia,a->i', l1_ee, cc.t1[q,:])
        vector1 -= 2*np.einsum('ijab,jab->i', l2_ee, cc.t2[q,:,:,:])
        vector1 +=   np.einsum('ijab,jba->i', l2_ee, cc.t2[q,:,:,:])

        vector2[q,:,:] += 2.*l1_ee
        vector2[:,q,:] -= l1_ee
        vector2 -= 2*np.einsum('ijab,a->ijb', l2_ee, cc.t1[q,:])
        vector2 +=   np.einsum('jiab,a->ijb', l2_ee, cc.t1[q,:])
    else:
        vector1 += l1_ee[:,q-nocc]
        vector2 += 2*l2_ee[:,:,q-nocc,:] - l2_ee[:,:,:,q-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def initial_2ppe_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=np.complex)
    return cc.amplitudes_to_vector_ip(vector1,vector2)
    
###################
# EE Greens       #
###################

def greens_b_vector_ee_rhf(cc,p,r):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=np.complex)
    if p < nocc:
        if r < nocc:
            vector1[r,:] -= cc.t1[p,:]
                       
            vector2[:,r,:,:] += cc.t2[p,:,:,:]
            vector2[r,:,:,:] -= 2*cc.t2[p,:,:,:]
            vector2[:,r,:,:] -= 2*cc.t2[p,:,:,:].transpose(0,2,1)
            vector2[r,:,:,:] += cc.t2[p,:,:,:].transpose(0,2,1)
            
            for i in range(nvir):
                vector2[r,r,i,i] *=0.5

        else:
            rp= r-nocc
            vector1 += 2*cc.t2[p,:,rp,:]
            vector1 -= cc.t2[:,p,rp,:]
            vector1 -= np.einsum('i,a->ia',cc.t1[:,rp],cc.t1[p,:])
            
            vector2 += 2*np.einsum('ija,b->ijab',cc.t2[:,:,rp,:],cc.t1[p,:])
            vector2 -= 4*np.einsum('ijb,a->ijab',cc.t2[:,:,rp,:],cc.t1[p,:])
            vector2 += 4*np.einsum('iab,j->ijab',cc.t2[p,:,:,:],cc.t1[:,rp])
            vector2 -= 2*np.einsum('jab,i->ijab',cc.t2[p,:,:,:],cc.t1[:,rp])
                        
            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5
    else:
        pp = p-nocc
        if r < nocc:
            vector1[r,pp] = 1.0
        else:
            rp = r-nocc
            vector1[:,pp] += cc.t1[:,rp] 
            
            vector2[:,:,pp,:] += 2*cc.t2[:,:,rp,:]
            vector2[:,:,:,pp] += 2*cc.t2[:,:,rp,:].transpose(1,0,2)
            vector2[:,:,pp,:] -= cc.t2[:,:,rp,:].transpose(1,0,2)
            vector2[:,:,:,pp] -= cc.t2[:,:,rp,:]
            
            for i in range(nocc):
                vector2[i,i,pp,pp] *= 0.5
            
    return cc.amplitudes_to_vector(vector1,vector2)
    
def greens_e0_ee_rhf(cc,q,s):
    nocc, nvir = cc.t1.shape
    if q < nocc:
        if s < nocc:
            e0 = -np.dot(cc.t1[q,:],cc.l1[s,:])
            e0 -= 2*np.dot(cc.t2[q,:,:,:].reshape(-1),cc.l2[s,:,:,:].reshape(-1))
            e0 += np.dot(cc.t2[q,:,:,:].reshape(-1),cc.l2[:,s,:,:].reshape(-1))
            
        else:
            sp = s - nocc
            e0 = 2*np.dot(cc.l1.reshape(-1),cc.t2[q,:,sp,:].reshape(-1))
            e0 -= np.dot(cc.l1.reshape(-1),cc.t2[:,q,sp,:].reshape(-1))
            e0 -= np.dot(cc.l1.reshape(-1),np.einsum('i,a->ia',cc.t1[:,sp],cc.t1[q,:]).reshape(-1))
            
            tmp = np.einsum('ija,b->ijab',cc.t2[:,:,sp,:],cc.t1[q,:])
            tmp -= 2*np.einsum('ijb,a->ijab',cc.t2[:,:,sp,:],cc.t1[q,:])
            tmp -= np.einsum('iab,j->ijab',cc.t2[q,:,:,:],cc.t1[:,sp])
            tmp += 2*np.einsum('jab,i->ijab',cc.t2[q,:,:,:],cc.t1[:,sp])
            e0 -= np.dot(cc.l2.reshape(-1),tmp.reshape(-1))
    else:
        qp = q - nocc
        if s < nocc:
            e0 = cc.l1[s,qp]
        else:
            sp = s - nocc
            e0 = np.dot(cc.t1[:,sp],cc.l1[:,qp])
            e0 += 2*np.dot(cc.t2[:,:,sp,:].reshape(-1),cc.l2[:,:,qp,:].reshape(-1))
            e0 -= np.dot(cc.t2[:,:,sp,:].reshape(-1),cc.l2[:,:,:,qp].reshape(-1))
    
    return e0

def greens_e_vector_ee_rhf(cc,q,s):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir))#,dtype=np.complex)
    vector2 = np.zeros((nocc,nocc,nvir,nvir))#,dtype=np.complex)
    if q < nocc:
        if s < nocc:
            vector1[q,:] -= cc.l1[s,:] #2
            vector1 += np.einsum('iab,b->ia', cc.l2[s,:,:,:], cc.t1[q,:]) #13
            vector1 -= 2*np.einsum('iab,b->ia', cc.l2[:,s,:,:], cc.t1[q,:]) #13
            
            vector2[:,q,:,:] += cc.l2[s,:,:,:]
            vector2[q,:,:,:] += cc.l2[s,:,:,:].transpose(0,2,1) #12
            vector2[:,q,:,:] -= 2*cc.l2[s,:,:,:].transpose(0,2,1)
            vector2[q,:,:,:] -= 2*cc.l2[s,:,:,:] #12
                        
            for j in range(nvir):
                vector2[q,q,j,j] *= 0.5
            
        else:
            sp = s-nocc
            
            vector1[q][sp] = 1.0 #1
            
            #These terms should vanish - the S vector must be a linear combination of excited states, which are all orthogonal to the left hand ground state.
            
            #vector1 += 2*cc.t1[q][sp]*cc.l1 #5
            #vector2 += 4*cc.l2*cc.t1[q][sp]
            #vector2 -= 2*cc.l2.transpose(0,1,3,2)*cc.t1[q][sp] #17
            
            vector1[q,:] -= np.einsum('ia,i->a', cc.l1, cc.t1[:,sp]) #6
            vector1[:,sp] -= np.einsum('ia,a->i', cc.l1, cc.t1[q,:]) #7
            
            vector1 += 2*np.einsum('ijab,jb->ia', cc.l2, cc.t2[q,:,sp,:]) #14
            vector1 -= np.einsum('ijab,jb->ia', cc.l2, cc.t2[q,:,:,sp])
            
            vector1[q,:] -= 2*np.einsum('ijab,ijb->a', cc.l2, cc.t2[:,:,sp,:])
            vector1[q,:] += np.einsum('ijab,ijb->a', cc.l2, cc.t2[:,:,:,sp]) #15
            
            vector1[:,sp] += np.einsum('ijab,jab->i', cc.l2, cc.t2[:,q,:,:]) #16
            vector1[:,sp] -= 2*np.einsum('ijab,jab->i', cc.l2, cc.t2[q,:,:,:])
            
            vector1 -= np.einsum('ijab,a,j->ib', cc.l2, cc.t1[q,:],cc.t1[:,sp])
            vector1 += 2*np.einsum('ijab,b,j->ia', cc.l2, cc.t1[q,:],cc.t1[:,sp])
            
            vector2[q,:,sp,:] += 2*cc.l1
            vector2[:,q,sp,:] -= cc.l1 #8    
            vector2[:,q,:,sp] += 2*cc.l1
            vector2[q,:,:,sp] -= cc.l1 #8
            
            vector2[:,q,:,:] -= 2*np.einsum('ijab,j->iab', cc.l2, cc.t1[:,sp]) #18
            vector2[q,:,:,:] += np.einsum('ijab,j->iab', cc.l2, cc.t1[:,sp])
            vector2[q,:,:,:] -= 2*np.einsum('ijab,j->iba', cc.l2, cc.t1[:,sp]) #18
            vector2[:,q,:,:] += np.einsum('ijab,j->iba', cc.l2, cc.t1[:,sp])
            
            vector2[:,:,sp,:] += 2*np.einsum('ijab,b->jia', cc.l2, cc.t1[q,:]) #19
            vector2[:,:,:,sp] -= np.einsum('ijab,b->jia', cc.l2, cc.t1[q,:])
            vector2[:,:,:,sp] += 2*np.einsum('ijab,b->ija', cc.l2, cc.t1[q,:]) #19
            vector2[:,:,sp,:] -= np.einsum('ijab,b->ija', cc.l2, cc.t1[q,:])
            
            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5
            
    else:
        qp = q-nocc
        if s < nocc:
            vector1 += 2*cc.l2[s,:,qp,:] #9
            vector1 -= cc.l2[s,:,:,qp]
            
        else:
            sp = s-nocc
            vector1[:,sp] += cc.l1[:,qp] #3
            vector1 += 2*np.einsum('ija,j->ia', cc.l2[:,:,:,qp], cc.t1[:,sp]) #11
            vector1 -= np.einsum('ija,j->ia', cc.l2[:,:,qp,:], cc.t1[:,sp])

            vector2[:,:,sp,:] -= cc.l2[:,:,qp,:].transpose(1,0,2)
            vector2[:,:,:,sp] -= cc.l2[:,:,qp,:]    
            vector2[:,:,sp,:] += 2*cc.l2[:,:,qp,:]  #10
            vector2[:,:,:,sp] += 2*cc.l2[:,:,qp,:].transpose(1,0,2)  #10  
            
            for i in range(nocc):
                vector2[i,i,sp,sp] *= 0.5       
    
    return cc.amplitudes_to_vector(vector1,vector2)

def initial_ee_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.ones((nocc,nvir),dtype=np.complex)
    vector2 = np.ones((nocc,nocc,nvir,nvir),dtype=np.complex)
    return cc.amplitudes_to_vector(vector1,vector2)
    
    
###################
# Greens Drivers  #
###################

class OneParticleGF:
    def __init__(self):
        self.gmres_tol = 1e-6

    def solve_ip(self,cc,ps,qs,omega_list,broadening):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print(" solving ip portion...")
        x0 = initial_ip_guess(cc)
        p0 = 0.0*x0 + 1.0
        e_vector = list() 
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc,q))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            b_vector = greens_b_vector_ip_rhf(cc,p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(cc.ipccsd_matvec, vector, curr_omega-1j*broadening)
                solver = gminres.gMinRes(matr_multiply,b_vector,x0,p0)
                sol = solver.get_solution().reshape(-1)
                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = -np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals
            
    def solve_2ppe(self,cc,ps,qs,omega_list,broadening,r1_ee,r2_ee,l1_ee,l2_ee,e):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print(" solving ip portion...")
        x0 = initial_2ppe_guess(cc)
        p0 = 0.0*x0 + 1.0
        n = len(x0)
        e_vector = list() 
        for q in qs:
            e_vector.append(greens_e_vector_2ppe_rhf(cc,q,l1_ee,l2_ee))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            b_vector = greens_b_vector_2ppe_rhf(cc,p,r1_ee,r2_ee)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(cc.ipccsd_matvec, vector, curr_omega-e-1j*broadening)
                
                counter = gmres_counter()
                H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
                sol, info = scipy.sparse.linalg.gmres(H,-b_vector, x0 = x0, restart = 100, tol = self.gmres_tol, callback = counter)
                print(info,counter.niter)
                    
                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def solve_ea(self,cc,ps,qs,omega_list,broadening):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print(" solving ea portion...")
        x0 = initial_ea_guess(cc)
        p0 = 0.0*x0 + 1.0
        e_vector = list() 
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(cc,p))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for iq,q in enumerate(qs):
            b_vector = greens_b_vector_ea_rhf(cc,q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(cc.eaccsd_matvec, vector, -curr_omega-1j*broadening)
                solver = gminres.gMinRes(matr_multiply,b_vector,x0,p0)
                sol = solver.get_solution().reshape(-1)
                x0 = sol
                for ip,p in enumerate(ps):
                    gfvals[ip,iq,iomega] = np.dot(e_vector[ip],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def kernel(self,cc,p,q,omega_list,broadening):
        return self.solve_ip(cc,p,q,omega_list,broadening), self.solve_ea(cc,p,q,omega_list,broadening)
    
    def solve_2pgf(self,cc,ps,qs,rs,ss,omega_list,broadening,dpq = None):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        if not isinstance(rs, collections.Iterable): rs = [rs]
        if not isinstance(ss, collections.Iterable): ss = [ss]

        nocc,nvir = cc.t1.shape
        x0 = initial_ee_guess(cc)
        p0 = 0.0*x0 + 1.0
        n = len(x0)
        #e_vector = np.zeros((len(qs),len(ss),n),dtype=np.complex)
        e_vectors = []
        e0s = np.zeros((len(qs),len(ss)))
        for nq,q in enumerate(qs):
            e_vectors.append([])
            for ns,s in enumerate(ss):
                e_vectors[nq].append([])
                if not (dpq is None) and (all(dpq[q,s,:] == 0)): continue
                e_vector = greens_e_vector_ee_rhf(cc,q,s)
                idx = np.flatnonzero(e_vector)
                val = e_vector[idx]
                e_vectors[nq][ns] = [idx,val]
                #e_vector[nq,ns,:] = greens_e_vector_ee_rhf(cc,q,s)
                e0s[nq,ns] = greens_e0_ee_rhf(cc,q,s)
        
        sys.stdout.flush()
  
        gfvals = np.zeros((len(ps),len(qs),len(rs),len(ss),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            for nr,r in enumerate(rs):
                
                if not (dpq is None) and (all(dpq[p,r,:] == 0)): continue
                print("Solve Linear System for p =",p,"r =",r)
                
                b_vector = greens_b_vector_ee_rhf(cc,p,r)

                for iomega in range(len(omega_list)):
                    curr_omega = omega_list[iomega]
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(cc.eomee_ccsd_matvec_singlet, vector, -curr_omega-1j*broadening)
                    print(iomega)
                    
                    counter = gmres_counter()
                    H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
                    #P = scipy.sparse.linalg.LinearOperator((n,n), matvec = precond)
                    sol, info = scipy.sparse.linalg.gmres(H,-b_vector, x0 = x0, restart = 150, tol = self.gmres_tol, callback = counter)
                    print(info,counter.niter)
                    
                    s0 = -np.dot(sol,cc.amplitudes_to_vector(cc.l1,cc.l2))
                    #s0 += 2*(p == r)*(r < nocc)
                    #s0 /= curr_omega+1j*broadening
                    
                    x0 = sol
                    for nq,q in enumerate(qs):
                       for ns,s in enumerate(ss):
                           
                           if dpq is None:
                               dot = 1.0
                           else:
                               dot = np.dot(dpq[p,r,:],dpq[q,s,:])
                           if (dot == 0): continue
                           
                           idx,val = e_vectors[nq][ns]
                           total = np.dot(val,sol[idx])
                           #total = np.dot(e_vector[nq,ns,:],sol)
                           total += e0s[nq,ns]*s0
                           gfvals[ip,nq,nr,ns,iomega] = total*dot
        
        if len(ps) == 1 and len(qs) == 1 and len(rs) == 1 and len(ss) == 1:
            return gfvals[0,0,0,0,:]
        else:
            return gfvals

class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
