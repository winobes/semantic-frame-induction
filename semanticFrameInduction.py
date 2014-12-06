import model0 as mod0
import numpy as np
import matplotlib.pyplot as plt

args = ('v','s','o')


def loadData( dataFile , trainPC, xvalidPC, testPC):

    counts = []
    word_to_index = {a: {} for a in args}
    index_to_word = {a: {} for a in args}
    V = {a: 0 for a in args}  
    trnData = {a: [] for a in args}

    xvData = []
    tstData= []
    # Load the 
    
    with open( dataFile) as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            c = int(c)
            splt = c*(trainPC/100)
    
            counts.append(c)
            for (w, a) in zip((v,s,o), args):
                if not w in word_to_index[a]: 
                    word_to_index[a][w] = V[a]
                    index_to_word[a][V[a]] = w
                    V[a] += 1
                trnData[a].append(word_to_index[a][w])
            splt2 = (c-splt)* (xvalidPC/100) 
            xvData.append((v,s,o, splt2))
            tstData.append((v,s,o, c-splt-splt2))
                
                                        
    return ((counts, word_to_index, index_to_word, V, trnData), xvData, tstData)
    

def runTests(*frames, *alphas):
    if len(args) = 0:
        frames = [5*(n+1) for n in range(10)]
        alphas = [1+0.1*n for n in range(11)]

     trnData, xvData, tstData = loadData("all_VSOs.sorted.concat",70,20,10)

    for a in alphas:
        for f in frames:
            mod0.em(f,a,trnData)

mod0.EM()
mod0
