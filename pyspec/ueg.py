import numpy as np
import time
import math
import sys
from operator import itemgetter

def sorter_function(inarr):
    val = inarr[ 0 ]*inarr[ 0 ] + inarr[ 1 ]*inarr[ 1 ] + inarr[ 2 ]*inarr[ 2 ]
    return val

def get_shell_degeneracy(nmax=512):
    nmax = int(nmax**(1./3))
    ijk = list()
    norm = list()
    for i in range(-nmax,nmax+1):
        for j in range(-nmax,nmax+1):
            for k in range(-nmax,nmax+1):
                ijk.append([i,j,k])
                norm.append(np.linalg.norm([i,j,k]))
    ijk = np.array(ijk)
    norm = np.array(norm)
    idx = norm.argsort()
    norm = norm[idx]
    ijk = ijk[idx,:]

    norms, degen = np.unique(norm, return_counts=True)
    shells = zip(norms, degen)
    return shells

def get_unique_orbitals(nbas):
    shells = get_shell_degeneracy()
    orbs = list()
    orb = 0
    orbs.append(orb)
    for shell in shells:
        norm, num = shell
        orb += num
        if orb >= nbas:
            break
        else:
            orbs.append(orb)
    return orbs

def change_basis_2el_complex(g,C):
    """Change basis for 2-el integrals with complex coefficients and return.

    - C is a matrix (Ns x Nnew) whose columns are new basis vectors,
      expressed in the basis in which g is given.
    - Typical operation is g in the site basis, and C is a 
      transformation from site to some-other-basis.
    """
    g1 = np.tensordot(C,g,axes=[0,3])
    g1 = np.transpose(g1,(1,2,3,0))
    # g1 is Ns x Ns x Ns x Nnew
    g = np.tensordot(C,g1,axes=[0,2])
    g = np.transpose(g,(1,2,0,3))
    # g is Ns x Ns x Nnew x Nnew
    g1 = np.tensordot(C.conj(),g,axes=[0,1])
    g1 = np.transpose(g1,(1,0,2,3))
    # g1 is Ns x Nnew x Nnew x Nnew
    g = np.tensordot(C.conj(),g1,axes=[0,0])
    # g is Nnew x Nnew x Nnew x Nnew
    return g

class UEG(object):
    def __init__(self, nelec, nbasis, rs, verbose=False):
        self.nelec = nelec
        # We limit ourselves to the closed shell case
        if ((self.nelec % 2) != 0):
            sys.exit("Need an even number of electrons!")
        self.nocc = self.nelec / 2
        self.nbas = nbasis
        self.dim = 3
        self.rs = rs
        self._ulim = 0
        self._llim = 0
        self._rgvecs = np.zeros(0)
        self._scf_en = 0.0
        self.verbose = verbose

        self._volume = self.nelec * (4.0/3.0) * np.pi * self.rs**3 
        self._length = self._volume ** (1./ 3.)
        self.madelung = 2.83729747948149 / self._length

        self.create_gvecs()
        if self.verbose: self.print_info()


    def create_gvecs(self):
        converged=False
        icur = int( math.ceil( ( 3. / ( 4. * np.pi ) * self.nbas ) ** ( 1. / 3. ) ) )
        Gvecs = []
        while not converged :
            curbas = 0
            self.gnorm = []
            if self.verbose: print "...finding gvecs in sphere ix = [ %4d ]" % icur
            for ix in range(-icur, icur+1):
                ix2 = ix*ix
                for iy in range(-icur, icur+1):
                    iy2 = iy*iy
                    for iz in range(-icur, icur+1):
                        iz2 = iz*iz
                        Gvecs.append(ix)
                        Gvecs.append(iy)
                        Gvecs.append(iz)
                        self.gnorm.append( ix2 + iy2 + iz2 )
                        curbas = curbas + 1
            self.gnorm = sorted( self.gnorm )
            if icur < np.sqrt( self.gnorm[ self.nbas - 1] ):
                converged = False
                icur = icur + 1
            else:
                converged = True
        upper_limit = self.nbas - 1
        same_value = True
        while same_value:
            upper_limit = upper_limit + 1
            if( self.gnorm[ upper_limit ] - self.gnorm[ upper_limit - 1 ] > 0 ):
                same_value = False
            else:
                same_value = True

        lower_limit = self.nbas
        same_value = True
        while same_value:
            lower_limit = lower_limit - 1
            if( self.gnorm[ lower_limit ] - self.gnorm[ lower_limit - 1 ] > 0 ):
                same_value = False
            else:
                same_value = True

        self.ulim = upper_limit
        self.llim = lower_limit
        outbas = self.ulim
        self.nbas = outbas
        self.rgvecs = np.zeros( ( outbas, 3 ) )
        count = 0
        maxnorm  = self.gnorm[ outbas - 1 ]
        for i in range( 0, int(len(Gvecs)/3) ):
            ix = Gvecs[ 3 * i + 0 ]
            iy = Gvecs[ 3 * i + 1 ]
            iz = Gvecs[ 3 * i + 2 ]
            if (ix*ix + iy*iy + iz*iz <= maxnorm ):
                self.rgvecs[ count ][ 0 ] = ix
                self.rgvecs[ count ][ 1 ] = iy
                self.rgvecs[ count ][ 2 ] = iz
                count = count + 1
        zrgvecs = self.rgvecs
        zrgvecs = sorted( zrgvecs, key=sorter_function )
        self.rgvecs = np.array( zrgvecs ).tolist()
        for i,j in enumerate(self.rgvecs): print i,j

    def print_k(self):
        twopidl = 2.0 * np.pi / self._length
        for i in range(0,len(self.gnorm)):
            print "K%3d : %14.8f" % (i,twopidl*np.sqrt(self.gnorm[ i ]))

    def kin(self, p, q):
        twopdg = 2. * np.pi / self._length
        twopdgsq = twopdg * twopdg
        kin = 0.0
        if p == q :
            px = self.rgvecs[p][0]
            py = self.rgvecs[p][1]
            pz = self.rgvecs[p][2]
            inorm = px*px + py*py + pz*pz
            kin = 0.5 * twopdgsq * inorm
        return kin

    def eri(self, p, q, r, s):
        twopdg = 2. * np.pi / self._length
        twopdgsq = twopdg * twopdg
        fac = 4.0 * np.pi / twopdgsq / self._volume

        pqx = self.rgvecs[p][0] - self.rgvecs[q][0]
        pqy = self.rgvecs[p][1] - self.rgvecs[q][1]
        pqz = self.rgvecs[p][2] - self.rgvecs[q][2]
        srx = self.rgvecs[s][0] - self.rgvecs[r][0]
        sry = self.rgvecs[s][1] - self.rgvecs[r][1]
        srz = self.rgvecs[s][2] - self.rgvecs[r][2]
        normsq = pqx*pqx + pqy*pqy + pqz*pqz

        integral = 0.0
        if( pqx == srx and pqy == sry and pqz == srz ):
            if normsq == 0 :
                integral = self.madelung
            else:
                integral = fac / (1.0*normsq)
        return integral

    def kohnsham(self):
        ks = np.zeros(self.nbas, )
        for p in range(0, self.nbas):
            ks[p] = self.kin(p, p) + self.vxc_lda()
        return ks 

    def vxc_lda(self):
        rs = self.rs
        density = 1.0 / (4./3 * np.pi * rs**3)

        A = 0.0311
        B = -0.048
        C = 0.0020
        D = -0.0116
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334

        vx = -(3.0*density/np.pi)**(1./3)
        if rs > 1:
            vc = gamma/(1+beta1*np.sqrt(rs)+beta2*rs)*( 
                      (1+7./6*beta1*np.sqrt(rs)+4./3*beta2*rs)
                    / (1+beta1*np.sqrt(rs)+beta2*rs) )
        else:
            vc = A*np.log(rs) + B - A/3. + 2./3*C*rs*np.log(rs) + (2*D-C)*rs/3

        return vx + vc

    #def energy(self):
    #    energy = 0.0
    #    for p in range( 0, self.nocc ):
    #        energy = energy + 2. * self.kin( p, p )
    #        for q in range( 0, self.nocc ):
    #            energy = energy - self.eri( q, p, p, q )
    #    self.scf_en = energy
    #    return energy

    def get_hcore(self):
        h = np.zeros((self.nbas, self.nbas))
        for p in range(self.nbas):
            h[p,p] = self.kin(p,p)
            #if p < self.nocc:
            #    h[p,p] -= self.madelung
        return h 

    def get_veff(self):
        # UEG doesn't have a Hartree energy due to positive background
        veff = np.zeros((self.nbas, self.nbas))
        for p in range(self.nbas):
            vk = 0.0
            for i in range(self.nocc):
                vk += self.eri(p,i,i,p)
            veff[p,p] = -vk
        return veff

    def get_fock(self):
        vcoul = np.zeros((self.nbas, self.nbas))
        hcore = self.get_hcore()
        for p in range(self.nbas):
            twoel = 0.0
            for i in range(self.nocc):
                twoel -= self.eri(p,i,i,p)
            vcoul[p,p] = twoel
        return hcore + vcoul

    def eri_chem_real(self):
        eri_chem = self.eri_full_fast()
        # Convert to physics <12|12>
        eri_phys = eri_chem.copy().transpose(0,2,1,3)

        if self.verbose:
            print "In complex PW basis"
            print "-------------------"
            print "Checking 4-fold symmetry:"
            print "<01|23> =? <10|32> :", np.allclose(eri_phys,eri_phys.transpose(1,0,3,2))
            
            print "Checking 8-fold symmetry:"
            print "<01|23> =? <03|21> :", np.allclose(eri_phys,eri_phys.transpose(0,3,2,1))

        gvecs = np.array(self.rgvecs)
        #print "Gvecs ="
        Umat = np.zeros((self.nbas,self.nbas), dtype=np.complex)
        basis = 0
        for gi,g in enumerate(gvecs):
            mgi = np.linalg.norm(-g-gvecs,axis=1).argmin()
            #print g, "-->", -g, "with index", mgi
            if gi == mgi:
                Umat[gi,basis] = 1.0
            elif gi < mgi:
                Umat[gi,basis] = 1.0/np.sqrt(2.0)
                Umat[mgi,basis] = 1.0/np.sqrt(2.0)
            else:
                Umat[gi,basis] = 1j/np.sqrt(2.0)
                Umat[mgi,basis] = -1j/np.sqrt(2.0)
            basis += 1

        # Change basis in fock (does nothing)
        #fock = np.dot(Umat.T.conj(),np.dot(np.diag(self.fock()),Umat)).real
        #print "self.fock() ="
        #print self.fock()
        #print "fock in new basis ="
        #print np.diag(fock)

        eri_phys_real = change_basis_2el_complex(eri_phys, Umat).real
        if self.verbose:
            print "In real cos/sin basis"
            print "---------------------"
            print "eri_phys_real[0,0,0,0] =", eri_phys_real[0,0,0,0]

            print "Checking 4-fold symmetry:"
            print "<01|23> =? <10|32> :", np.allclose(eri_phys_real,eri_phys_real.transpose(1,0,3,2))
            
            print "Checking 8-fold symmetry:"
            print "<01|23> =? <03|21> :", np.allclose(eri_phys_real,eri_phys_real.transpose(0,3,2,1))

        self._eri_chem_real = eri_phys_real.copy().transpose(0,2,1,3)
        return self._eri_chem_real

    def eri_full_slow(self):
        start = time.clock()
        outvec = np.zeros((self.nbas, self.nbas,
                           self.nbas, self.nbas))
        twopdg = 2. * np.pi / self._length
        twopdgsq = twopdg * twopdg
        fac = 4.0 * np.pi / twopdgsq / self._volume
        integral = 0.0
        for p in range( 0, len( self.rgvecs ) ):
            px = self.rgvecs[ p ][ 0 ]
            py = self.rgvecs[ p ][ 1 ]
            pz = self.rgvecs[ p ][ 2 ]
            for q in range( 0, len( self.rgvecs ) ):
                pqx = px - self.rgvecs[ q ][ 0 ]
                pqy = py - self.rgvecs[ q ][ 1 ]
                pqz = pz - self.rgvecs[ q ][ 2 ]
                normsq = pqx * pqx + pqy * pqy + pqz * pqz
                if normsq == 0 :
                    integral = self.madelung
                else:
                    integral = fac / (1.0*normsq)
                for r in range( 0, len( self.rgvecs ) ):
                    rx = self.rgvecs[ r ][ 0 ]
                    ry = self.rgvecs[ r ][ 1 ]
                    rz = self.rgvecs[ r ][ 2 ]
                    for s in range( 0, len( self.rgvecs ) ):
                        srx = self.rgvecs[ s ][ 0 ] - rx
                        sry = self.rgvecs[ s ][ 1 ] - ry
                        srz = self.rgvecs[ s ][ 2 ] - rz
                        if( pqx == srx and
                            pqy == sry and
                            pqz == srz ):
                            outvec[ p, q, r, s ] = integral

        #for p in range( 0, len( self.rgvecs ) ):
        #    for q in range( 0, len( self.rgvecs ) ):
        #        for r in range( 0, len( self.rgvecs ) ):
        #            for s in range( 0, len( self.rgvecs ) ):
        #                if( outvec[ p, q, r, s ] > 1e-6 ):
        #                    print "%3d %3d %3d %3d = %14.8f " % ( p, q, r, s, outvec[ p, q, r, s ] )
        end = time.clock()
        if self.verbose: print "SLOW TIME : %14.8f" % (end-start)
        return outvec

    def eri_full_fast(self):
        start = time.clock()
        outvec = np.zeros((self.nbas, self.nbas,
                           self.nbas, self.nbas))
        twopdg = 2. * np.pi / self._length
        twopdgsq = twopdg * twopdg
        fac = 4.0 * np.pi / twopdgsq / self._volume
        integral = 0.0
        for p in range( 0, len( self.rgvecs ) ):
            px = self.rgvecs[ p ][ 0 ]
            py = self.rgvecs[ p ][ 1 ]
            pz = self.rgvecs[ p ][ 2 ]
            for q in range( p, len( self.rgvecs ) ):
                pqx = px - self.rgvecs[ q ][ 0 ]
                pqy = py - self.rgvecs[ q ][ 1 ]
                pqz = pz - self.rgvecs[ q ][ 2 ]
                normsq = pqx*pqx + pqy*pqy + pqz*pqz
                if normsq == 0 :
                    integral = self.madelung
                else:
                    integral = fac / (1.0*normsq)
                for r in range( p, len( self.rgvecs ) ):
                    rx = self.rgvecs[ r ][ 0 ]
                    ry = self.rgvecs[ r ][ 1 ]
                    rz = self.rgvecs[ r ][ 2 ]
                    for s in range( 0, len( self.rgvecs ) ):
                        srx = self.rgvecs[ s ][ 0 ] - rx
                        sry = self.rgvecs[ s ][ 1 ] - ry
                        srz = self.rgvecs[ s ][ 2 ] - rz
                        if( pqx == srx and
                            pqy == sry and
                            pqz == srz ):
                            outvec[ p, q, r, s ] = integral
                            outvec[ q, p, s, r ] = integral
                            outvec[ r, s, p, q ] = integral
                            outvec[ s, r, q, p ] = integral

        end = time.clock()
        if self.verbose: print "FAST TIME : %14.8f" % (end-start)
        #for p in range( 0, len( self.rgvecs ) ):
        #    for q in range( 0, len( self.rgvecs ) ):
        #        for r in range( 0, len( self.rgvecs ) ):
        #            for s in range( 0, len( self.rgvecs ) ):
        #                if( outvec[ p, q, r, s ] > 1e-6 ):
        #                    print "%3d %3d %3d %3d = %14.8f " % ( p, q, r, s, outvec[ p, q, r, s ] )
        return outvec

    def print_info(self):
        print "  ::      Uniform Electron Gas Parameters     :: "
        print "    Dimension of System       = %14d " % self.dim
        print "    Number of Electrons       = %14d " % self.nelec
        print "    Madelung constant         = %14.8f " % self.madelung
        print "    Volume of Box             = %14.8f " % self._volume
        print "    Length of Box             = %14.8f " % self._length
        print "    Number of Basis Functions = %14d " % self.nbas

def main():
    #ueg = UEG(114, 485, 4.0, True)
    ueg = UEG(14, 19, 4.0, True)
    #fock_elements = ueg.fock()
    ueg.print_k()
    twoel = ueg.eri_full_fast()

if __name__ == '__main__':
    main()
