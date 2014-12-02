import numpy as np
import model0 as mod0
import random as rnd

def frame_coherency(K):
    coh = 0
    (frame_dists, frame_assign, theta, data) = mod0.em(10,1.5) 
    probs = { }
    for i in range(K):
        ((v,s,o),f) = frame_assign.popitem()
        print(v,s,o,f)
        probs[(v,s,o)] = frame_dists[f]['v'][v]*frame_dists[f]['s'][s]*frame_dists[f]['o'][o]*theta[f]

    N = len(data)
    for tup in probs:
        vR,_,_ = data[rnd.randint(0,N)]
        probR = np.argmax([frame_dists[frm]['v'][vR]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists])
        if probs[tup] > probR:
            coh +=1

    print(" accuracy:", coh/K)
   

frame_coherency(1000)
