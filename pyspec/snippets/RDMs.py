#   Do the ground state CCSD calculation
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-8
    e_cc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    nocc, nvir = t1.shape
    ntot = nocc+nvir

#   Calculate the diagonal of the RDM for each excited state         
    tot = 0.0
    for i in range(ntot):
        occ_i = gf.rdm(mycc,i,i)
        print i,occ_i
        tot += occ_i
    print tot
	
#   Calculate and biorthonormalise the right and left eigenvectors
    EEnroots = 10
    e_ee, r_ee = mycc.eomee_ccsd_singlet(EEnroots)
    e_ee_l, l_ee = mycc.eomee_ccsd_singlet(EEnroots, left = True)
    r_ee, l_ee = mycc.biorthonormalize(r_ee,l_ee,e_ee,e_ee_l)

#   Calculate the diagonal of the RDM for each excited state     
	tot = np.zeros(EEnroots)
    for i in range(ntot):
        occ_i = gf.e_rdm(mycc,i,i,r_ee,l_ee)
        print i,occ_i
        tot += occ_i
    print tot