import numpy as np

def polarizability(mol, mf, td, comega, ao_dipole=None): 
    mol.set_common_orig((0.0,0.0,0.0))
    if ao_dipole is None:
        ao_dipole = mol.intor_symmetric('int1e_r', comp=3) 
    occidx = np.where(mf.mo_occ==2)[0] 
    viridx = np.where(mf.mo_occ==0)[0] 
    mo_coeff = mf.mo_coeff 
    orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx] 
    virocc_dip = np.einsum('xaq,qi->xai', np.einsum('pa,xpq->xaq', orbv, ao_dipole), orbo) 
    p = np.zeros((comega.size,3), dtype=np.complex128) 
    for (x,y),e in zip(td.xy, td.e): 
        dip = np.einsum('xai,ai->x',virocc_dip, np.sqrt(2.0)*(x+y))
        for iw,w in enumerate(comega): 
            p[iw] += dip**2*((1.0/(w-e))-(1.0/(w+e))) 
    return p

def main():
    from pyscf import gto, scf, dft, tddft

    mol = gto.M(atom='H 0.00 0.76 -0.93; H 0.00 -0.76 -0.93; O 0.0 0.0 -0.35', basis='631g')
    #mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.verbose = 0
    mf.kernel()

    rpa = tddft.TDHF(mf) 
    rpa.nstates = 10
    rpa.kernel()

    comegas = np.linspace(0.0, 1.0, num=300)+1j*0.01
    spec = polarizability(mol, mf, rpa, comegas)
    for w, s in zip(comegas, spec):
        print w.real, s[0].real, s[0].imag, s[1].real, s[1].imag, s[2].real, s[2].imag

if __name__ == '__main__':
    main()

