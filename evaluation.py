import numpy as np
import model0 as mod0
import random as rnd
import matplotlib.pyplot as pp

args = ('v','s','o')

def frames_by_frequency(frame_dists):

    frame_freq = {f: list(frame_dists[f].items()) for f in frame_dists}
    for f in frame_freq:
        frame_freq[f].sort(key=lambda x: x[1], reverse=True)
    return frame_freq

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


def frame_coherency(model, data):

    # trained model
    if len(model) == 3:
        (frame_dists, frame_assign, theta) = model
        M = 0
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
    verbs = []
    for (v,s,o) in data.keys():
        for i in range(data[(v,s,o)]):
            verbs.append(v)

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


    #probsR = [frame_dists[ frame_assign[(rV,s,o)]]['v'][rV]*frame_dists[frame_assign[(rV,s,o)]]['s'][s]*frame_dists[frame_assign[(rV,s,o)]]['o'][o]*theta[frame_assign[(rV,s,o)]] for (rV,v,s,o) in tstD]
    #probsR = [np.argmax([frame_dists[frm]['v'][rV]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists]) for (rV,s,o) in tstD]
        
    for i in range(N):
        #(v,s,o) = dataList[i]
        c = data[(v,s,o)]
        total += c
        if probs[i] > probsR[i]:
            coh += c
      
    return (coh/total)


def top_verbs( x , model, N ):

    nrOfFrms = len(frame_dists)
    if x == 0:
        (frame_assign, frame_dists, theta) = model
        topVs = {frm: [(_,v) for t in [ frame_dists[frm]['v'][:N ] for frm in range(nrOfFrms)]]}


    return topVs
            
                
