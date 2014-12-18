import numpy as np
import ctypes
from random import randrange
from probfuncs import normalize

gibbs_c = ctypes.CDLL('./gibbs_c.so').gibbs

args = ('v', 's', 'o')

def gibbs(F, alpha, beta, T, inData):

    sentence_count = inData
    N = len(inData)

    # build translation maps
    V = 0
    verb_to_index = {}
    index_to_verb = {}
    W = 0
    word_to_index = {}
    index_to_word = {}
    for (v,s,o) in inData:
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
        c, randrange(F)] for ((v,s,o),c) in inData.items()], dtype=np.long)


    gibbs_c(ctypes.c_void_p(arrData.ctypes.data), ctypes.c_int(F), ctypes.c_int(T),
            ctypes.c_double(alpha), ctypes.c_double(beta),
            ctypes.c_long(N), ctypes.c_long(V), ctypes.c_long(W))

    frame_assign = {}
    frame_count = {f: 0 for f in range(F)}
    for (v, s, o, c, f) in arrData:
        v = index_to_verb[v]
        s = index_to_word[s]
        o = index_to_word[o]
        assert sentence_count[(v,s,o)] == c # sanity check
        frame_assign[(v,s,o)] = f
        frame_count[f] += c
    
    # infer the frame prior from assignments
    theta = normalize([frame_count[f] for f in range(F)])
    # infer frame's argument distributons from assignments
    frame_dists = construct_frame_dists(frame_assign, F, sentence_count, frame_count)

    return (frame_dists, frame_assign, theta)

# questionable.
def construct_frame_dists(frame_assign, F, counts, frame_count):
    totals_in_frame = {f: {a: 0 for a in args} for f in range(F)}
    frame_dists = {f: {a: {} for a in args} for f in range(F)}
    for ((v,s,o), f) in frame_assign.items():
        for (a,w) in zip(args, (v,s,o)):
            if w in frame_dists[f][a]:
                frame_dists[f][a][w] += counts[(v,s,o)]
            else:
                frame_dists[f][a][w] = counts[(v,s,o)]
    # normalize
    for f in frame_dists:
        for a in args:
            for w in frame_dists[f][a]:
                frame_dists[f][a][w] /= frame_count[f]
    return frame_dists

