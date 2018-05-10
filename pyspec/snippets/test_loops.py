# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:11:15 2018

@author: Bryan Lau
"""

import pis_hf
import time
import numpy as np

dpq = pis_hf.dip_mo(2/5.29e-11,4)[1]
ntot = dpq.shape[1]
dpq = dpq.reshape(ntot,ntot,3)
omegas = 10
EEgf = np.random.random((ntot,ntot,ntot,ntot,omegas))

print('Testing loops of Eegf summations!, ntot = ',ntot)
print('Size of EEgf',EEgf.size,EEgf.shape)

# Alan is a FORtran man!
spectrum = np.zeros(omegas)
start_time = time.time()
for p in range(ntot):
    for r in range(ntot):
        if not (all(dpq[p,r,:] == 0)):
            for q in range(ntot):
                for s in range(ntot):
                    spectrum -= EEgf[p,q,r,s,:]
print('Naive: ',time.time() - start_time)

# replace all with np.all - we know dpq is a ndarray, and avoid the overhead of
# the python function - although that's not really showed in timeit tests...
'''
%timeit all(dpq[p,r,:] == 0)
1.9 µs ± 8.53 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

%timeit (dpq[p,r,:] == 0).all()
2.43 µs ± 21.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit np.all(dpq[p,r,:] == 0)
3.41 µs ± 19.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
'''
spectrum2 = np.zeros(omegas)
start_time = time.time()
for p in range(ntot):
    for r in range(ntot):
        if not (dpq[p,r,:] == 0).all():
            for q in range(ntot):
                for s in range(ntot):
                    spectrum2 -= EEgf[p,q,r,s,:]
print('Numpy all optimization: ',time.time() - start_time)
print('Same as naive loop?',np.allclose(spectrum2,spectrum))

# even faster: replace all with any, directly checks for non-zeros without
# a boolean output comparison first
'''
%timeit dpq[p,r,:].any()
1.63 µs ± 18.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
'''
spectrum3 = np.zeros(omegas)
start_time = time.time()
for p in range(ntot):
    for r in range(ntot):
        if dpq[p,r,:].any():
            for q in range(ntot):
                for s in range(ntot):
                    spectrum3 -= EEgf[p,q,r,s,:]
print('Numpy any optimization: ',time.time() - start_time)
print('Same as naive loop?',np.allclose(spectrum3,spectrum))

# replace inner loop with sums
spectrum4 = np.zeros(omegas)
start_time = time.time()
for p in range(ntot):
    for r in range(ntot):
        if dpq[p,r,:].any():
            spectrum4 -= EEgf[p,:,r,:,:].sum(axis=(0,1))
print('Inner loop elimination: ',time.time() - start_time)
print('Same as naive loop?',np.allclose(spectrum4,spectrum))

# finally, full numpy call. Condition eegf, then sum it. I assume the real EEgf
# has zeros where dpq is zero, so I do not include it in the timing loop.
for p in range(ntot):
    for r in range(ntot):
        if not dpq[p,r,:].any():
            EEgf[p,:,r,:,:] = 0
start_time = time.time()
spectrum5 = -EEgf.sum(axis=(0,1,2,3))
print('Full sum: ',time.time() - start_time)
print('Same as naive loop?',np.allclose(spectrum5,spectrum))

'''
Final output:

Testing loops of Eegf summations!, ntot =  116
Size of EEgf 1810639360 (116, 116, 116, 116, 10)
Naive:  28.242925882339478
Numpy all optimization:  28.92085075378418
Same as naive loop? True
Numpy any optimization:  29.17117953300476
Same as naive loop? True
Inner loop elimination:  0.4436802864074707
Same as naive loop? True
Full sum:  4.108199596405029
Same as naive loop? True

On my personal machine, I noticed that the numpy all/any optimizations performed
faster than the python all().    
'''