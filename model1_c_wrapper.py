import numpy as np
import ctypes
from random import randrange
from probfuncs import normalize

gibbs_c = ctypes.CDLL('./gibbs_c.so').gibbs

args = ('v', 's', 'o')

def gibbs(F, alpha, beta, T, burnIn, data):

    listData = list(data.items())
    N = len(data)

    V = 0
    verb_to_index = {}
    W = 0
    word_to_index = {}
    for (v,s,o) in data:
        if not v in verb_to_index:
            verb_to_index[v] = V
            V += 1
        if not s in word_to_index:
            word_to_index[s] = W
            W += 1
        if not o in word_to_index:
            word_to_index[o] = W
            W += 1

    # turn the data into an array
    arrData = np.array([[verb_to_index[v],word_to_index[s],word_to_index[o],
        c, randrange(F)] for ((v,s,o),c) in listData], dtype=np.long)

    # set up the array to storte the samples in
    arrSamples = np.array([[0 for t in range(F)] for _ in range(N)], dtype=np.int)

    gibbs_c(ctypes.c_void_p(arrData.ctypes.data), ctypes.c_void_p(arrSamples.ctypes.data),
            ctypes.c_long(N), ctypes.c_long(V), ctypes.c_long(W), ctypes.c_int(F), ctypes.c_int(T),
            ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_int(burnIn))

    samples = {(v,s,o): {f: int(arrSamples[i][f]) for f in range(F)} 
            for i, ((v,s,o),_) in enumerate(listData)}

    return samples 

