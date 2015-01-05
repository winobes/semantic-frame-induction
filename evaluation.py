import numpy as np
import dicesim
import random as rnd
import matplotlib.pyplot as pp


args = ('v','s','o')

def frames_by_frequency(frame_dists):

    frame_freq = {f: list(frame_dists[f].items()) for f in frame_dists}
    for f in frame_freq:
        frame_freq[f].sort(key=lambda x: x[1], reverse=True)
    return frame_freq

def m0_to_dists(m0):
    frame_dists = m0[0]
    F = len(frame_dists)
    frame_dists_v = {f:{} for f in range(F)}
    frame_dists_w = {f:{} for f in range(F)}
    for f in range(F):
        for v in frame_dists[f]['v']:
            frame_dists_v[f][v] = frame_dists[f]['v'][v]
        for s in frame_dists[f]['s']:
            frame_dists_w[f][s] = frame_dists[f]['s'][s]/2
        for o in frame_dists[f]['o']:
            if o in frame_dists_w[f]:
                frame_dists_w[f][o] += frame_dists[f]['o'][o]/2
            else:
                frame_dists_w[f][o] = frame_dists[f]['o'][o]/2
    return(frame_dists_v, frame_dists_w)


def show_most_common(frame_dists_v, frame_dists_w, top=25):

    frame_freq_v = frames_by_frequency(frame_dists_v)
    frame_freq_w = frames_by_frequency(frame_dists_w)

    stopwords = ['it', 'you', 'them', 'him', 'he', 'i', 'they', 'we', 'me', 'us', 'she', 'her', 'me'] 

    for f in frame_freq_v:
        print('----------- frame', f,'-------------')
        print('verbs\t\t\tsubjects/objects')
        j = 0
        for i in range(top):
            if frame_freq_v[f][i][1] < 0.001:
                print(end='\t\t\t')
            else:
                tablen = ((7 + len(frame_freq_v[f][i][0])) // 8)
                tabs = '\t' if tablen >= 2 else '\t\t'
                print("%.4f"%frame_freq_v[f][i][1], frame_freq_v[f][i][0], end=tabs)
            while frame_freq_w[f][j][0] in stopwords:
                j += 1
            tablen = ((7 + len(frame_freq_w[f][j][0])) // 8)
            tabs = '\t' if tablen >= 2 else '\t\t'
            print("%.4f"%frame_freq_w[f][j][1], frame_freq_w[f][j][0], end=tabs)
            print()
            j += 1
        print()

# frame_coherency: calculates the maximal probability of a tuple and compares it with
# the maximal probability on the model of same tuple in which the verb is replaced by a random verb
def frame_coherency(model, data):

    # determine which model to evaluate 
    if len(model) == 3:
        (frame_dists, frame_assign, theta) = model
        M = 0
    #if model1: set `frame_assign'
    elif len(model) == 2:
        (verb_dists, word_dists) = model
        frame_assign  =  {(v,s,o): np.argmax([verb_dists[f][v]*word_dists[f][s]*word_dists[f][o] for f in verb_dists],axis=0) for (v,s,o) in data}
        M = 1
        
    # initialize variables
    total = 0            # actual number of tested examples (unique entries in model * counts)
    coh = 0              # number of tuples with higher probability on model than with random verb
    N = len(data)

    # lists to store results
    probs = []
    probsR = []

    # calculate max probability of each tuple in testset (=data)    
    if M == 0:
        probs = [frame_dists[frame_assign[(v,s,o)]]['v'][v]*frame_dists[frame_assign[(v,s,o)]]['s'][s]*frame_dists[frame_assign[(v,s,o)]]['o'][o]*theta[frame_assign[(v,s,o)]] for(v,s,o) in data ]
    elif M == 1:
        probs = []
        for (v,s,o) in data:
            frm = frame_assign[(v,s,o)]
            probs.append(verb_dists[frm][v]*word_dists[frm][s]*word_dists[frm][o])
        #verbDists = {f: {index_to_verb[i]: verbDists[i][f] for i in range(V)} for f in range(F)}

    # initialize tuples with verb replaced by random verb
    #     first: list all verbs ( with repitition )
    verbs = []
    for (v,s,o) in data.keys():
        for i in range(data[(v,s,o)]):
            verbs.append(v)

    # second: create tuples with some random verb
    tstD = [(rnd.choice(verbs), vso[0], vso[1], vso[2]) for vso in data]

    # calculate max probability of tuple with random verb on model
    for (rV,v,s,o) in tstD:
         if (rV,s,o) in frame_assign: #frame_assign.has_key():
             frm = frame_assign[(rV,s,o)]
             if M == 0:
                 probsR.append(frame_dists[frm]['v'][rV]*frame_dists[frm]['s'][s]* frame_dists[frm]['o'][o]*theta[frm])
             elif M == 1:
                 probsR.append(verb_dists[frm][rV]*word_dists[frm][s]*word_dists[frm][o])
         else:
             probsR.append(0)

# ----------- other possibilities for probabilities of tuples with random verb
    #probsR = [frame_dists[ frame_assign[(rV,s,o)]]['v'][rV]*frame_dists[frame_assign[(rV,s,o)]]['s'][s]*frame_dists[frame_assign[(rV,s,o)]]['o'][o]*theta[frame_assign[(rV,s,o)]] for (rV,v,s,o) in tstD]
    #probsR = [np.argmax([frame_dists[frm]['v'][rV]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists]) for (rV,s,o) in tstD]
# ----------------------------------------------------------------------------

    # summations of datapoints to calculate percentage of higher scoring datapoints 
    for i in range(N):
        c = data[(v,s,o)]
        total += c
        if probs[i] > probsR[i]:
            coh += c
      
    return (coh/total)

# calculate average dicesim score of model
#     use verbs per frame which have probability higher than `cutOffP'
#     use max `fn_threshold' per frame verbs from framenet 
def frame_accuracy(verb_dists,cutOffP,fn_threshold): 

    # build list of lists of verbs that are assigned to a frame
    mod_sets = []
    for frm in verb_dists:
        modS = []
        for verb in verb_dists[frm]:
            if verb_dists[frm][verb] > cutOffP:
                modS.append(verb)
        mod_sets.append(modS)
        
    # retrieve lists of framenet verbs
    fn_sets = dicesim.framenet.sort_verbs_to_frames(fn_threshold)

    # calculate and return average dicesim score (over frames) for this model
    best = 0
    tot = 0
    for modS in mod_sets:
        for fnS in fn_sets.values():
            modS = set(modS)
            fnS = set(fnS)
            cur = (( 2 * len(modS & fnS) ) / ( len(modS) + len(fnS) ))
            if cur > best:
                best = cur
        tot += best

    return tot/len(mod_sets)

