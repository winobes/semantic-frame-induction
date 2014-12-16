import numpy as np
import model0 as mod0
import random as rnd
import matplotlib.pyplot as pp

def frame_coherency(model, data):

    (frame_dists, frame_assign, theta) = model
    results = []

    total = 0
    coh = 0
    N = len(data)
    probs = []
    probsR = []

    dataList = list(data.keys())

    
    probs = [frame_dists[frame_assign[(v,s,o)]]['v'][v]*frame_dists[frame_assign[(v,s,o)]]['s'][s]*frame_dists[frame_assign[(v,s,o)]]['o'][o]*theta[frame_assign[(v,s,o)]] for(v,s,o) in data ]
   
    tstD = [(rnd.choice(dataList)[0], vso[0], vso[1], vso[2]) for vso in data]

    for (rV,v,s,o) in tstD:
         if (rV,s,o) in frame_assign: #frame_assign.has_key():
             probsR.append(frame_dists[ frame_assign[(rV,s,o)]]['v'][rV]*frame_dists[frame_assign[(rV,s,o)]]['s'][s]* frame_dists[frame_assign[(rV,s,o)]]['o'][o]*theta[frame_assign[(rV,s,o)]])
         else:
             probsR.append(0)


    #probsR = [frame_dists[ frame_assign[(rV,s,o)]]['v'][rV]*frame_dists[frame_assign[(rV,s,o)]]['s'][s]*frame_dists[frame_assign[(rV,s,o)]]['o'][o]*theta[frame_assign[(rV,s,o)]] for (rV,v,s,o) in tstD]
    #probsR = [np.argmax([frame_dists[frm]['v'][rV]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists]) for (rV,s,o) in tstD]
    
    

    for i in range(N):
        (v,s,o) = dataList[i]
        c = data[(v,s,o)]
        total += c
        if probs[i] > probsR[i]:
            coh += c
    
    """"
    for tup in probs:
        vR,_,_,_ = data[rnd.randint(0,N-1)]
        probR = np.argmax([frame_dists[frm]['v'][vR]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists])
        if probs[tup] > probR:
            coh +=1
    """

    return (coh/total)
    
