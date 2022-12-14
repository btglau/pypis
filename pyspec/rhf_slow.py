#!/usr/bin/env python
#
# Author: Timothy Berkelbach 
# Modified by: Bryan Lau

'''
Slow explicit TDHF/TDH
'''

from functools import reduce
import time
import tempfile
import numpy
import numpy as np
import h5py
import ao2mo_custom
from scipy import linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.tddft import rhf

einsum = lib.einsum

class TDHF(rhf.TDHF):
    def __init__(self, mf):
        rhf.TDHF.__init__(self,mf)
        self.frozen = 0 

##################################################
# don't modify the following attributes, they are not input options
        self.eris = None
        self._nocc = None
        self._nmo = None
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        elif isinstance(self.frozen, (int, numpy.integer)):
            return int(self.mo_occ.sum()) // 2 - self.frozen
        elif self.frozen:
            occ_idx = self.mo_occ > 0
            occ_idx[numpy.asarray(self.frozen)] = False
            return numpy.count_nonzero(occ_idx)
        else:
            return int(self.mo_occ.sum()) // 2

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        if isinstance(self.frozen, (int, numpy.integer)):
            return len(self.mo_occ) - self.frozen
        else:
            return len(self.mo_occ) - len(self.frozen)

    def kernel(self, x0=None):
        self.eris = self.ao2mo()
        return rhf.TDHF.kernel(self, x0)
    
    def direct(self):
        self.eris = self.ao2mo()
        
        log = logger.Logger(self._scf.stdout, self._scf.verbose)
        log.info('\n')
        log.info('******** {} for {} ********'.format(
                 self.__class__, self._scf.__class__))
        log.info('    Diagonalize the full A/B Hamiltonian')
        
        vind,hdiag = self.gen_vind(self._scf)
        cput0 = time.clock(), time.time()
        H = vind(np.identity(hdiag.size))
        log.timer('Build full A/B H',*cput0)
        
        w,v = linalg.eig(H)        
        # We only need positive eigenvalues
        realidx = np.where((abs(w.imag) < 1e-6) & (w.real > 0))[0]
        idx = realidx[w[realidx].real.argsort()]
        w,v = w[idx].real, v[:,idx].real
      
        def norm_xy(z):
            nocc = self.nocc
            nvir = self.nmo - nocc
            x, y = z.reshape(2,nvir,nocc)
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2) # normalize to 0.5 for alpha spin
            norm = 1/np.sqrt(norm)
            return x*norm, y*norm
        
        # list comprehensions grab by row, so transpose V
        self.xy = [norm_xy(z) for z in v.T]
        self.e = w
        return
    
    def ao2mo(self):
        if hasattr(self.eris,'ovov'):
            return self.eris
        return _ERIS(self)

    def get_ab_mats(self, mf):
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        eris = self.eris
        nov = nocc*nvir
        a_mat = np.zeros((nov,nov))
        b_mat = np.zeros((nov,nov))
        ai = 0
        for i in range(nocc):
            for a in range(nvir):
                a_mat[ai,ai] = fvv[a,a] - foo[i,i]
                bj = 0
                for j in range(nocc):
                    for b in range(nvir):
                        a_mat[ai,bj] += 2*eris.ovvo[i,a,b,j] - eris.oovv[j,i,a,b]
                        b_mat[ai,bj] += 2*eris.ovov[i,a,j,b] - eris.ovov[j,a,i,b]
                        bj += 1
                ai += 1

        return a_mat, b_mat        

    def gen_vind(self, mf):
        singlet = self.singlet

        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        hdiag2 = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

        eris = self.eris
        #a_mat, b_mat = self.get_ab_mats(mf)
        #rpa_mat = np.bmat([[a_mat, b_mat], [-b_mat, -a_mat]])

        def vind(xys):
            # xys is (nz, 2*nocc*nvir)
            nz = len(xys)
            hx = numpy.empty((nz,2,nvir,nocc))
            #hx = numpy.empty((nz,2,nvir*nocc))
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                hx[i,0] = hdiag*x - einsum('jiab,bj->ai', eris.oovv, x) \
                            - einsum('jaib,bj->ai', eris.ovov, y)
                hx[i,1] = - (hdiag*y - einsum('jiab,bj->ai', eris.oovv, y) \
                            - einsum('jaib,bj->ai', eris.ovov, x))
                if singlet:
                    hx[i,0] += 2*einsum('iabj,bj->ai', eris.ovvo, x) \
                             + 2*einsum('iajb,bj->ai', eris.ovov, y)
                    hx[i,1] += - (2*einsum('iabj,bj->ai', eris.ovvo, y) \
                                + 2*einsum('iajb,bj->ai', eris.ovov, x))

                #x, y = xys[i].reshape(2,nvir*nocc)
                #hx[i,0] = np.dot(a_mat, x) + np.dot(b_mat, y)
                #hx[i,1] = - np.dot(b_mat, x) - np.dot(a_mat, y)
            return hx.reshape(nz,-1)

        return vind, hdiag2
    

class TDH(TDHF):

    def get_ab_mats(self, mf):
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)

        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        eris = self.eris
        nov = nocc*nvir
        a_mat = np.zeros((nov,nov))
        b_mat = np.zeros((nov,nov))
        ai = 0
        for a in range(nvir):
            for i in range(nocc):
                a_mat[ai,ai] = fvv[a,a] - foo[i,i]
                bj = 0
                for b in range(nvir):
                    for j in range(nocc):
                        a_mat[ai,bj] += 2*eris.ovvo[i,a,b,j]
                        b_mat[ai,bj] += 2*eris.ovov[i,a,j,b]
                        bj += 1
                ai += 1

        return a_mat, b_mat        

    def gen_vind(self, mf):
        singlet = self.singlet

        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        hdiag2 = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

        eris = self.eris
        #a_mat, b_mat = self.get_ab_mats(mf)
        #rpa_mat = np.bmat([[a_mat, b_mat], [-b_mat, -a_mat]])

        def vind(xys):
            # xys is (nz, 2*nocc*nvir)
            nz = len(xys)
            #hx = numpy.empty((nz,2,nvir*nocc))
            hx = numpy.empty((nz,2,nvir,nocc))
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                hx[i,0] = hdiag*x
                hx[i,1] = -hdiag*y
                if singlet:
                    hx[i,0] += 2*einsum('iabj,bj->ai', eris.ovvo, x) \
                             + 2*einsum('iajb,bj->ai', eris.ovov, y)
                    hx[i,1] += - (2*einsum('iabj,bj->ai', eris.ovvo, y) \
                                + 2*einsum('iajb,bj->ai', eris.ovov, x))
                #x, y = xys[i].reshape(2,nvir*nocc)
                #hx[i,0] = np.dot(a_mat, x) + np.dot(b_mat, y)
                #hx[i,1] = - np.dot(b_mat, x) - np.dot(a_mat, y)
            return hx.reshape(nz,-1)

        return vind, hdiag2


class TDHTDA(rhf.TDA):
    def __init__(self, mf):
        rhf.TDA.__init__(self,mf)
        self.frozen = 0 

##################################################
# don't modify the following attributes, they are not input options
        self.eris = None
        self._nocc = None
        self._nmo = None
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        elif isinstance(self.frozen, (int, numpy.integer)):
            return int(self.mo_occ.sum()) // 2 - self.frozen
        elif self.frozen:
            occ_idx = self.mo_occ > 0
            occ_idx[numpy.asarray(self.frozen)] = False
            return numpy.count_nonzero(occ_idx)
        else:
            return int(self.mo_occ.sum()) // 2

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        if isinstance(self.frozen, (int, numpy.integer)):
            return len(self.mo_occ) - self.frozen
        else:
            return len(self.mo_occ) - len(self.frozen)

    def kernel(self, x0=None):
        self.eris = self.ao2mo()
        return rhf.TDA.kernel(self, x0)
    
    def direct(self):
        self.eris = self.ao2mo()
        
        log = logger.Logger(self._scf.stdout, self._scf.verbose)
        log.info('\n')
        log.info('******** {} for {} ********'.format(
                 self.__class__, self._scf.__class__))
        log.info('    Diagonalize the full A/- Hamiltonian')
        
        vind,hdiag = self.gen_vind(self._scf)
        cput0 = time.clock(), time.time()
        H = vind(np.identity(hdiag.size))
        log.timer('Build full A/- H',*cput0)
        
        w,v = linalg.eig(H)        
        # We only need positive eigenvalues
        realidx = np.where((abs(w.imag) < 1e-6) & (w.real > 0))[0]
        idx = realidx[w[realidx].real.argsort()]
        w,v = w[idx].real, v[:,idx].real
        
        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # list comprehensions grab by row, so transpose V
        self.xy = [(xi.reshape(nvir,nocc)*numpy.sqrt(.5),0) for xi in v.T]
        self.e = w
        return

    def ao2mo(self):
        if hasattr(self.eris,'ovov'):
            return self.eris
        return _ERIS(self)

    def gen_vind(self, mf):
        singlet = self.singlet

        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        hdiag2 = hdiag.ravel()

        eris = self.eris

        def vind(zs):
            # zs is (nz, nocc*nvir)
            nz = len(zs)
            hx = numpy.empty((nz,nvir,nocc))
            for i,z in enumerate(zs):
                z = z.reshape(nvir,nocc)
                hx[i] = hdiag*z
                if singlet:
                    hx[i] += 2*einsum('iabj,bj->ai', eris.ovvo, z)
            return hx.reshape(nz,-1)

        return vind, hdiag2

class TDA(TDHTDA):        
    def gen_vind(self, mf):
        singlet = self.singlet
        
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])

        hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        hdiag2 = hdiag.ravel()

        eris = self.eris
        
        def vind(zs):
            # zs is (nz, nocc*nvir)
            nz = len(zs)
            hx = numpy.empty((nz,nvir,nocc))
            for i,z in enumerate(zs):
                z = z.reshape(nvir,nocc)
                hx[i] = hdiag*z - einsum('jiab,bj->ai', eris.oovv, z)
                if singlet:
                    hx[i] += 2*einsum('iabj,bj->ai', eris.ovvo, z)

            return hx.reshape(nz,-1)

        return vind, hdiag2

RPA = TDHF
dRPA = TDH
dTDA = TDHTDA
CIS = TDA

def _mem_usage(nocc, nvir):
    # super compact form, transforming only the integrals needed
    incore = nocc*(nocc+nvir)*(nocc+nvir)*(nocc+nvir+1)/2
    # Roughly, factor of two for safety 
    incore *= 2
    eris = 3*nocc**2*nvir**2
    # size of H + eigenvectors
    H = 2*(nocc*nvir)**2
    return incore*8/1e6, H*8/1e6, eris*8/1e6


class _ERIS:
    '''Almost identical to gw ERIS except only oovv, ovov, ovvo.
    
    'cc' is because this code was pulled from ccsd module
    '''
    def __init__(self, cc, mo_coeff=None):
        log = logger.Logger(cc.stdout, cc.verbose)
        log.info('\n')
        log.info('******** {} for {} ********'.format(self.__class__,cc._scf.__class__))
        log.info('   ao2mo custom integrals ovov oovv ovvo')
        
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))
        self.dtype = mo_coeff.dtype

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        orbo = mo_coeff[:,:nocc]
        mo_list = (orbo,mo_coeff,mo_coeff,mo_coeff)
        
        mem_incore, mem_h, mem_eris = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]
        log.info('   Mem tot, used, avail: {} {} {}'.format(int(cc.max_memory),int(mem_now),int(cc.max_memory-mem_now)))
        log.info('   Mem needed, incore: {}, 3*oovv type: {}, H+ev: {}'.format(int(mem_incore),int(mem_eris),int(mem_h)))
        log.info('   Mem after transform + H + evec: {}'.format(int(mem_now-mem_incore-mem_eris-mem_h)))
        
        if mem_now + mem_incore < cc.max_memory:
            log.info('   Transform in memory')
            eri = ao2mo.incore.general(cc._scf._eri, mo_list)
            
            if lib.current_memory()[0] + mem_eris > cc.max_memory:
                log.info('   Delete AOs to make space for oovv type eris')
                cc._scf._eri = None
            
            self.ovov = numpy.empty((nocc,nvir,nocc,nvir))
            self.oovv = numpy.empty((nocc,nocc,nvir,nvir))
            self.ovvo = numpy.empty((nocc,nvir,nvir,nocc))
            buf = numpy.empty((nmo,nmo,nmo))
            for i in range(nocc):
                lib.unpack_tril(eri[i*nmo:(i+1)*nmo], out=buf)
                self.ovov[i] = buf[nocc:,:nocc,nocc:]
                self.oovv[i] = buf[:nocc,nocc:,nocc:]
                self.ovvo[i] = buf[nocc:,nocc:,:nocc]
            buf = None
            eri = None
            
            if lib.current_memory()[0] + mem_h > cc.max_memory:
                log.info('   Delete AOs to make space for H + evec')
                cc._scf._eri = None

        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            raise NotImplementedError

        else:
            if mem_eris > cc.max_memory:
                log.info('   Transform to disk. MO remains on disk, AO deleted')
                ds_type = mo_coeff.dtype.char
                self.feri = lib.H5TmpFile()
                self.ovov = self.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
                self.oovv = self.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
                self.ovvo = self.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), ds_type)
            else:
                log.info('   Transform to disk. MO loaded back to memory, AO deleted')
                self.ovov = numpy.empty((nocc,nvir,nocc,nvir))
                self.oovv = numpy.empty((nocc,nocc,nvir,nvir))
                self.ovvo = numpy.empty((nocc,nvir,nvir,nocc))

            tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo_custom.general(cc._scf, mo_list, tmpfile2.name, 'aa')
            cc._scf._eri = None
            cput1 = time.clock(), time.time()
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmo,nmo,nmo))
                for i in range(nocc):
                    lib.unpack_tril(f['aa'][i*nmo:(i+1)*nmo], out=buf)
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovvo[i] = buf[nocc:,nocc:,:nocc]
                del(f['aa'])
                buf = None
            cput1 = log.timer_debug1('gather ovov, oovv, ovvo', *cput1)

        log.timer('TDSCF integral transformation', *cput0)
    
if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    td = rhf.TDHF(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309283  33.84489902  33.84489902]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440704  32.380853    33.03572939]

    td = TDHF(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309283  33.84489902  33.84489902]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440704  32.380853    33.03572939]

    td = TDH(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 23.44384642  23.44384642  27.98402844  48.41758942  48.41758942]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 22.87728841  22.87728841  25.81461147  47.56131188  47.56131188]

    td = TDHTDA(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 23.46396995  23.46396995  28.2123058   48.43105188  48.43105188]
