import numpy as np
import model0 as mod0
import random as rnd
import matplotlib.pyplot as pp

def frame_coherency(model, data):

    (frame_dists, frame_assign, theta) = model
    results = []

    coh = 0
    N = len(data)
    probs = []
    probsR= []
    #    for (v,s,o,c) in data:

    #print("************\n frame_assign len: \t%d\n data len: %d\n************\n" %(len(frame_assign),N))

    
    probs = [frame_dists[frame_assign[(v,s,o)]]['v'][v]*frame_dists[frame_assign[(v,s,o)]]['s'][s]*frame_dists[frame_assign[(v,s,o)]]['o'][o]*theta[frame_assign[(v,s,o)]] for(v,s,o,_) in data ]
    print("tst*",end="", flush=True)     
    
    tstD = [(data[rnd.randint(0,N-1)][0],data[i][0],data[i][1],data[i][2]) for i in range(N) ]
    
    probsR = [frame_dists[ frame_assign[(v,s,o)]]['v'][rV]*frame_dists[frame_assign[(v,s,o)]]['s'][s]*frame_dists[frame_assign[(v,s,o)]]['o'][o]*theta[frame_assign[(v,s,o)]] for (rV,v,s,o) in tstD]
    #probsR = [np.argmax([frame_dists[frm]['v'][rV]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists]) for (rV,s,o) in tstD]
    print("tst*",end="", flush=True)     
    

    
    for i in range(N):
        if probs[i] > probsR[i]:
            coh += 1
    
    """"
    for tup in probs:
        vR,_,_,_ = data[rnd.randint(0,N-1)]
        probR = np.argmax([frame_dists[frm]['v'][vR]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists])
        if probs[tup] > probR:
            coh +=1
    """

    return (coh/len(data))
    
