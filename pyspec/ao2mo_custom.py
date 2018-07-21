#!/usr/bin/env python
#
# Author: Bryan Lau
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:05:22 2018

@author: Bryan Lau


A module that will do on-disk transformation of two electron integrals, and 
also return specific slices of (o)ccupied and (v)irtual ones needed for post HF
"""

import numpy, h5py, tempfile

from pyscf import lib
from pyscf.lib import logger

IOBLK_SIZE = 128
f8_size = numpy.dtype('f8').itemsize/1024**2

def general(mf, mo_coeffs, erifile, dataname='eri_mo',
                ioblk_size=IOBLK_SIZE,
                compact=True):
    '''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals on the fly.
    Args:
        eri : 8-fold reduced eri vector
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str or h5py File or h5py Group object
            To store the transformed integrals, in HDF5 format.
    Kwargs
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals
    '''
    log = logger.Logger(mf.stdout, mf.verbose)
    log.info('******** ao2mo disk, custom eri ********')
    eri = mf._eri
    
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]
    
    nao_pair = nao*(nao+1) // 2
    if compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
        ij_red = False
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        ij_red = True
        nij_pair = nmoi*nmoj
    if compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3]):
        kl_red = False
        nkl_pair = nmok*(nmok+1) // 2
    else:
        kl_red = True
        nkl_pair = nmok*nmol
    
    chunks_half = (numpy.minimum(int(IOBLK_SIZE//(nao_pair*f8_size)),nmoj),
                   numpy.minimum(int(IOBLK_SIZE//(nij_pair*f8_size)),nmol))
    chunks_full = (numpy.minimum(int(IOBLK_SIZE//(nkl_pair*f8_size)),nmoj),
                   numpy.minimum(int(IOBLK_SIZE//(nij_pair*f8_size)),nmol))
    
    if isinstance(erifile, str):
        if h5py.is_hdf5(erifile):
            feri = h5py.File(erifile)
            if dataname in feri:
                del(feri[dataname])
        else:
            feri = h5py.File(erifile,'w',libver='latest')
    else:
        assert(isinstance(erifile, h5py.Group))
        feri = erifile
    h5d_eri = feri.create_dataset(dataname,(nij_pair,nkl_pair),'f8',chunks=chunks_full) 
    
    tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    feri_swap = h5py.File(tmpfile2.name,'w',libver='latest')
    half_eri = feri_swap.create_dataset(dataname,(nij_pair,nao_pair),'f8',chunks=chunks_half)
    
    log.debug('Memory information:')
    log.debug('  IOBLK_SIZE (MB): {}'.format(IOBLK_SIZE))
    log.debug('  Final disk eri size (MB): {:.3g}, chunked {:.3g}'
              .format(nij_pair*nkl_pair*f8_size,numpy.prod(chunks_full)*f8_size))
    log.debug('  Half transformed eri size (MB): {:.3g}, chunked {:.3g}'
              .format(nij_pair*nao_pair*f8_size,numpy.prod(chunks_half)*f8_size))
    log.debug('  RAM buffer for half transform (MB): {:.3g}'
             .format(nij_pair*chunks_half[1]*f8_size*2))
    log.debug('  RAM buffer for full transform (MB): {:.3g}'
             .format(f8_size*chunks_full[0]*nkl_pair*2 + chunks_half[0]*nao_pair*f8_size*2))
    
    def save1(piece,buf):
        start = piece*chunks_half[1]
        stop = (piece+1)*chunks_half[1]
        if stop > nao_pair:
            stop = nao_pair
        half_eri[:,start:stop] = buf[:,:stop-start]
        return
    
    def load2(piece):
        start = piece*chunks_half[0]
        stop = (piece+1)*chunks_half[0]
        if stop > nij_pair:
            stop = nij_pair
            if start >= nij_pair:
                start = stop - 1
        return half_eri[start:stop,:]
    
    def prefetch2(piece):
        start = piece*chunks_half[0]
        stop = (piece+1)*chunks_half[0]
        if stop > nij_pair:
            stop = nij_pair
            if start >= nij_pair:
                start = stop - 1
        buf_prefetch[:stop-start,:] = half_eri[start:stop,:]
        return
    
    def save2(piece,buf):
        start = piece*chunks_full[0]
        stop = (piece+1)*chunks_full[0]
        if stop > nij_pair:
            stop = nij_pair
        h5d_eri[start:stop,:] = buf[:stop-start,:]
        return
    
    # transform \mu\nu -> ij
    cput0 = time.clock(), time.time()
    Cimu = mo_coeffs[0].conj().transpose()
    buf_write = numpy.empty((nij_pair,chunks_half[1]))
    buf_out = numpy.empty_like(buf_write)
    wpiece = 0
    with lib.call_in_background(save1) as async_write:
        for lo in range(nao_pair):
            if lo % chunks_half[1] == 0 and lo > 0:
                #save1(wpiece,buf_write)
                buf_out, buf_write = buf_write, buf_out
                async_write(wpiece,buf_out)
                wpiece += 1
            buf = lib.unpack_row(eri,lo)
            uv = lib.unpack_tril(buf)
            uv = Cimu @ uv @ mo_coeffs[1]
            if ij_red:
                ij = numpy.ravel(uv) # grabs by row
            else:
                ij = lib.pack_tril(uv)
            buf_write[:,lo % chunks_half[1]] = ij
    # final write operation & cleanup
    save1(wpiece,buf_write)
    log.timer('(uv|lo) -> (ij|lo)', *cput0)
    uv = None
    ij = None
    buf = None

    # transform \lambda\sigma -> kl
    cput1 = time.clock(), time.time()
    Cklam = mo_coeffs[2].conj().transpose()
    buf_write = numpy.empty((chunks_full[0],nkl_pair))
    buf_out = numpy.empty_like(buf_write)
    buf_read = numpy.empty((chunks_half[0],nao_pair))
    buf_prefetch = numpy.empty_like(buf_read)
    rpiece = 0
    wpiece = 0
    with lib.call_in_background(save2,prefetch2) as (async_write,prefetch):
        buf_read = load2(rpiece)
        prefetch(rpiece+1)
        for ij in range(nij_pair):
            if ij % chunks_full[0] == 0 and ij > 0:
                #save2(wpiece,buf_write)
                buf_out, buf_write = buf_write, buf_out
                async_write(wpiece,buf_out)
                wpiece += 1
            if ij % chunks_half[0] == 0 and ij > 0:
                #buf_read = load2(rpiece)
                buf_read, buf_prefetch = buf_prefetch, buf_read
                rpiece += 1
                prefetch(rpiece+1)
            lo = lib.unpack_tril(buf_read[ij % chunks_half[0],:])
            lo = Cklam @ lo @ mo_coeffs[3]
            if kl_red:
                kl = numpy.ravel(lo)
            else:
                kl = lib.pack_tril(lo)
            buf_write[ij % chunks_full[0],:] = kl
    save2(wpiece,buf_write)
    log.timer('(ij|lo) -> (ij|kl)', *cput1)
    
    feri_swap.close()
    if isinstance(erifile, str):
        feri.close()
    return erifile

def incore_custom(eri, mo_coeffs, compact=True):
    '''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals incore. Capability of reduced transformation.
    
    This function is far worse than the pyscf native implementation that actually
    supports transforming only a subset (ao2mo general)
    
    Args:
        eri : 8-fold reduced eri vector
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
    Kwargs
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals
    '''
    log = logger.Logger(mf.stdout, mf.verbose)
    log.info('******** ao2mo in memory, custom eri ********')
    
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]
    
    nao_pair = nao*(nao+1) // 2
    if compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
        ij_red = False
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        ij_red = True
        nij_pair = nmoi*nmoj
    if compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3]):
        nkl_pair = nmok*(nmok+1) // 2
    else:
        nkl_pair = nmok*nmol
    
    log.debug('Memory information:')
    log.debug('  Final eri size (MB): {:.3g}'.format(nij_pair*nkl_pair*f8_size))
    log.debug('  Half transformed eri size (MB): {:.3g}'.format(nij_pair*nao_pair*f8_size))
    
    half_e1 = numpy.empty((nij_pair,nao_pair))
    feri = numpy.empty((nij_pair,nkl_pair))
    
    cput0 = time.clock(), time.time()
    Ciu = mo_coeffs[0].conj().transpose()
    for lo in range(nao_pair):
        buf = lib.unpack_row(eri,lo)
        uv = lib.unpack_tril(buf)
        uv = Ciu @ uv @ mo_coeffs[1]
        if ij_red:
            ij = numpy.ravel(uv) # grabs by row; i X j
        else:
            ij = lib.pack_tril(uv)
        half_e1[:,lo] = ij
    log.timer('(uv|lo) -> (ij|lo)', *cput0)
    buf = None
    uv = None
    ij = None

    cput1 = time.clock(), time.time()
    Cklam = mo_coeffs[2].conj().transpose()
    for ij in range(nij_pair):
        lo = lib.unpack_tril(half_e1[ij,:])
        lo = Cklam @ lo @ mo_coeffs[3]
        kl = lib.pack_tril(lo)
        feri[ij,:] = kl
    log.timer('(ij|lo) -> (ij|kl)', *cput1)
    lo = None
    kl = None
    
    return feri

def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

if __name__ == '__main__':
    import pis_hf, time
    from pyscf import ao2mo
    
    # start the calculation
    mol,mf,args = pis_hf.init_pis()
    
    # HF
    mf.kernel()
    mo_coeff = mf.mo_coeff
    
    # compare custom outcore eri with incore eri
    nmo = args.norb
    nocc = numpy.count_nonzero(mf.mo_occ)
    nvir = nmo - nocc
    
    print('\n\nIncore transformation (pyscf, full)...')
    start_time = time.time()
    eri_incore = ao2mo.incore.full(mf._eri, mo_coeff)
    eri_full = ao2mo.restore(1, eri_incore, nmo)
    ovov_incore = eri_full[:nocc,nocc:,:nocc,nocc:].copy()
    oovv_incore = eri_full[:nocc,:nocc,nocc:,nocc:].copy()
    ovvo_incore = eri_full[:nocc,nocc:,nocc:,:nocc].copy()
    print('Time elapsed (s): ',time.time() - start_time)
    eri_full = None
    
    print('\n\nIncore transformation (pyscf, compact)...')
    start_time = time.time()
    orbo = mo_coeff[:,:nocc]
    eri_incore2 = ao2mo.incore.general(mf._eri, (orbo,mo_coeff,mo_coeff,mo_coeff))
    ovov_incore2 = numpy.empty((nocc,nvir,nocc,nvir))
    oovv_incore2 = numpy.empty((nocc,nocc,nvir,nvir))
    ovvo_incore2 = numpy.empty((nocc,nvir,nvir,nocc))
    buf = numpy.empty((nmo,nmo,nmo))
    for i in range(nocc):
        lib.unpack_tril(eri_incore2[i*nmo:(i+1)*nmo], out=buf)
        ovov_incore2[i] = buf[nocc:,:nocc,nocc:]
        oovv_incore2[i] = buf[:nocc,nocc:,nocc:]
        ovvo_incore2[i] = buf[nocc:,nocc:,:nocc]
    buf = None
    print('Time elapsed (s): ',time.time() - start_time)
    
    print('\n\nOutcore transformation...')
    ds_type = mo_coeff.dtype.char
    feri = lib.H5TmpFile()
    ovov_outcore = feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
    oovv_outcore = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
    ovvo_outcore = feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), ds_type)
    tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    orbo = mo_coeff[:,:nocc]
    general(mf, (orbo,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa')
    outcore_time = time.time()
    with h5py.File(tmpfile2.name) as f:
        buf = numpy.empty((nmo,nmo,nmo))
        for i in range(nocc):
            lib.unpack_tril(f['aa'][i*nmo:(i+1)*nmo], out=buf)
            ovov_outcore[i] = buf[nocc:,:nocc,nocc:]
            oovv_outcore[i] = buf[:nocc,nocc:,nocc:]
            ovvo_outcore[i] = buf[nocc:,nocc:,:nocc]
        buf = None
        print('Reduced Incore (pyscf) vs outcore (custom)?',numpy.allclose(eri_incore2,f['aa']))
        del(f['aa'])
    print('Extract MOs: ',time.time() - outcore_time)
        
    print('\n\novov incore pyscf vs outcore custom',numpy.allclose(ovov_incore,ovov_outcore))
    print('oovv incore pyscf vs outcore custom',numpy.allclose(oovv_incore,oovv_outcore))
    print('ovvo incore pyscf vs outcore custom',numpy.allclose(ovvo_incore,ovvo_outcore))
    
    print('ovov pyscf full vs general',numpy.allclose(ovov_incore,ovov_incore2))
    print('oovv pyscf full vs general',numpy.allclose(oovv_incore,oovv_incore2))
    print('ovvo pyscf full vs general',numpy.allclose(ovvo_incore,ovvo_incore2))
    
    print('\n\nFull outcore transformation...')
    general(mf, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa')
    with h5py.File(tmpfile2.name) as f:   
        print('Incore (pyscf) vs outcore (custom)?',numpy.allclose(eri_incore,f['aa']))
        #print('Incore (custom) vs outcore (custom)?',numpy.allclose(eri_incore3,f['aa']))     
        del(f['aa'])
    
    # close hdf5 eri
    tmpfile2.close()