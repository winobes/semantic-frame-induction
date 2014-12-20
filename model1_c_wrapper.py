import numpy as np
import ctypes
from random import randrange
import csv
from probfuncs import increment

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
    data = np.array([[verb_to_index[v],word_to_index[s],word_to_index[o],
        sentence_count[(v,s,o)], randrange(F)] for (v,s,o) in sentence_count], dtype=np.long)
   
    # aliases of the array columns
    verbs    = data[:,0]
    subjects = data[:,1]
    objects  = data[:,2]
    counts   = data[:,3]

    # set up the array to store the samples in
    samples = np.zeros((N,F), dtype=np.int32)

    gibbs_c(ctypes.c_void_p(data.ctypes.data), ctypes.c_void_p(samples.ctypes.data),
            ctypes.c_long(N), ctypes.c_long(V), ctypes.c_long(W), ctypes.c_int(F), ctypes.c_int(T),
            ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_int(burnIn))
   
    samples *= counts.reshape((N,1))
    frameTotals = samples.sum(axis=0)

    verbDists =  np.zeros((V, F))
    verbDists += np.apply_along_axis(lambda x: np.bincount(verbs, x, minlength=V), 0, samples)
    verbDists /= frameTotals

    wordDists =  np.zeros((W, F))
    wordDists += np.apply_along_axis(lambda x: np.bincount(subjects, x, minlength=W), 0, samples)
    wordDists += np.apply_along_axis(lambda x: np.bincount(objects, x, minlength=W), 0, samples)
    wordDists /= 2 * frameTotals

    verbDists = {f: {index_to_verb[i]: verbDists[i][f] for i in range(V)} for f in range(F)}
    wordDists = {f: {index_to_word[i]: wordDists[i][f] for i in range(W)} for f in range(F)}

    return(verbDists, wordDists) 

def samples_to_dists(samples, counts, F):

    frame_count_v = {f:{} for f in range(F)}
    frame_count_w = {f:{} for f in range(F)}
    frame_count   = {f: 0 for f in range(F)}
    for (v,s,o) in samples:
        c = counts[(v,s,o)]
        for f in range(F):
            gibbs_hits = samples[(v,s,o)][f] 
            increment(frame_count, f, amount = c * gibbs_hits)
            increment(frame_count_v[f], v, amount = c * gibbs_hits)
            increment(frame_count_w[f], s, amount = c * gibbs_hits)
            increment(frame_count_w[f], o, amount = c * gibbs_hits)
    frame_dist_v = {f: {v: frame_count_v[f][v] / frame_count[f] 
        for v in frame_count_v} for f in range(F)}
    frame_dist_w = {f: {w: frame_count_v[f][v] / frame_count[f]
        for w in frame_count_w} for f in range(F)}

    return (frame_dist_v, frame_dist_w)

def save_dists(dists, filename):
    print("Saving frame dists to ", filename)
    F = len(dists)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for word in samples:
            writer.writerow([word] + [dists[f][word] if word in dists[f] else 0 for f in range(F)])

def read_dists(filename):
    dists = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader: 
            for f,p in enumerate(row[1:]):
                dists[f][row[0]] = p
    return dists

def save_samples(samples, filename):
    print("Saving samples to ", filename)
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
