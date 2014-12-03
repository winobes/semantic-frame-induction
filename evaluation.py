import numpy as np
import model0 as mod0
import random as rnd
import matplotlib.pyplot as pp

def frame_coherency(K):

    results = []
    frmsL = [5*(n+1) for n in range(10)]
    for frms in frmsL:
        coh = 0
        (frame_dists, frame_assign, theta, data) = mod0.em(frms,1.5) 
        N = len(data)
        probs = { }
        for i in range(K):
            ((v,s,o),f) = frame_assign.popitem()
            print(v,s,o,f)
            probs[(v,s,o)] = frame_dists[f]['v'][v]*frame_dists[f]['s'][s]*frame_dists[f]['o'][o]*theta[f]

        for tup in probs:
            vR,_,_ = data[rnd.randint(0,N)]
            probR = np.argmax([frame_dists[frm]['v'][vR]*frame_dists[frm]['s'][s]*frame_dists[frm]['o'][o]*theta[frm] for frm in frame_dists])
            if probs[tup] > probR:
                coh +=1

        results.append(coh/K)

    pp.plot(frmsL,results)
    pp.show()

frame_coherency(1000 )
