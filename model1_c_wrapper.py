import numpy as np
import ctypes
from random import randrange
import csv

gibbs_c = ctypes.CDLL('./gibbs_c.so').gibbs

args = ('v', 's', 'o')

def gibbs(F, alpha, beta, T, burnIn, sentence_count):

    N = len(sentence_count)

    # build translation maps
    V = 0
    verb_to_index = {}
    index_to_verb = {}
    W = 0
    word_to_index = {}
    index_to_word = {}
    for (v,s,o) in sentence_count:
        if not v in verb_to_index:
            verb_to_index[v] = V
            index_to_verb[V] = v
            V += 1
        if not s in word_to_index:
            word_to_index[s] = W
            index_to_word[W] = s
            W += 1
        if not o in word_to_index:
            word_to_index[o] = W
            index_to_word[W] = o
            W += 1

    # turn the data into an array
    arrData = np.array([[verb_to_index[v],word_to_index[s],word_to_index[o],
        sentence_count[(v,s,o)], randrange(F)] for (v,s,o) in sentence_count], dtype=np.long)

    # set up the array to storte the samples in
    arrSamples = np.zeros((N,F), dtype=np.long)

    gibbs_c(ctypes.c_void_p(arrData.ctypes.data), ctypes.c_void_p(arrSamples.ctypes.data),
            ctypes.c_long(N), ctypes.c_long(V), ctypes.c_long(W), ctypes.c_int(F), ctypes.c_int(T),
            ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_int(burnIn))

    samples = {}
    for i in range(N):
        v = index_to_verb[arrData[i][0]]
        s = index_to_word[arrData[i][1]]
        o = index_to_word[arrData[i][2]]
        samples[(v,s,o)] = list(arrSamples[i])
        if any(x > T for x in arrSamples[i]):
            print((v,s,o), arrSamples[i])

    return samples 

def save_samples(samples, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for (v,s,o) in samples:
            writer.writerow([v,s,o] + samples[(v,s,o)])

def read_samples(filename):
    samples = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            samples[tuple(row[:3])] = [int(c) for c in row[3:]]
    return samples

