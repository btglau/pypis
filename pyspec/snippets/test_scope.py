# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:40:42 2018

@author: Bryan Lau
"""

import numpy

a = numpy.random.random((5,5))
b = numpy.random.random((5,5))

def hi(asdf):
   asdf[0,0] = b[0,0]
    
print(a)
print(b)
hi(a)
print(a[0,0])