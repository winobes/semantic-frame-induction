import model0 as mod0
import evaluation as evalu
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import pickle
import collections
import os
from probfuncs import weighted_sample
import itertools

args = ('v','s','o')

def pruneData( dataFile='Preprocessing/all_VSOs.sorted.concat', cutoffPC=1.5):
    print('Pruning to %', cutoffPC, '... ', end='')
    V  = {}
    SO = {}
    data = {}
    with open(dataFile) as f:
        # build the verb vocabulary
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            c = int(c)
            if v in V:
                V[v] += c
            else: 
                V[v] = c
            if s in SO:
                SO[s] += c
            else:
                SO[s] = c
            if o in SO:
                SO[o] += c
            else:
                SO[o] = c
            if (v,s,o) in data:
                data[(v,s,o)] += c
            else:
                data[(v,s,o)] = c

    # determine the cutoffs
    countsV = list(V.values())
    countsV.sort(reverse=True)

    countsSO = list(SO.values())
    countsSO.sort(reverse=True)

    cutoffV = countsV[int(len(countsV) * cutoffPC/100)-1]
    cutoffSO = countsSO[int(len(countsSO) * cutoffPC/100)-1]

    print("The absolute cutoff for verb frequency will be", cutoffV)
    print("The absolute cutoff for subject/object frequency will be", cutoffSO)
    # pop off the data with low counts
    for (v,s,o) in data.copy():
        if V[v] < cutoffV or SO[s] < cutoffSO or SO[o] < cutoffSO:
            data.pop((v,s,o))

    pickle.dump(data, open("allData.pkl", 'wb'))
    print('Done.')

def splitData( dataFile='allData.pkl', testPC=10, xvPC=20):

    # load up all the data
    if not os.path.isfile(dataFile):
        raise ValueError ("must run pruneData first.")
    data = pickle.load(open(dataFile, 'rb'))
 
    # put it all in a big long list and shuffle
    dataList = list(itertools.chain(*([vso for i in range(data[vso])] for vso in data)))
    rnd.shuffle(dataList)

    print("Splitting off %", testPC, "for testing data and %", xvPC, "for cross verb data...")
    # split accordingly
    N = len(dataList)
    split1 = int(N*testPC/100)
    split2 = split1 + int(N*xvPC/100)
    print("Total of ", N, "items. 0 to", split1, "to testList,", split1, "to", split2, "to xvList.")
    testList       = dataList[:split1]
    xvList         = dataList[split1:split2]
    trainingList   = dataList[split2:]

    # make the lists into multisets
    def to_multiset(bigList):
        multiset = {}
        for item in bigList:
            if item in multiset:
                multiset[item] += 1
            else:
                multiset[item] = 1
        return multiset
    trainingData = to_multiset(trainingList)
    testData = to_multiset(testList)
    xvData = to_multiset(xvList)

    # check that we didn't make any mistakes...
    def merge_counts(d1, d2):
        d3 = d1.copy()
        for val in d2:
            if val in d1:
                d3[val] += d2[val]
            else:
                d3[val] = d2[val]
        return d3
    assert data == merge_counts(trainingData, merge_counts(testData, xvData))

    # dump it!
    pickle.dump(trainingData, open("trainingData.pkl", 'wb'))
    pickle.dump(testData, open("testingData.pkl", 'wb'))
    pickle.dump(xvData, open("xvData.pkl", 'wb'))

def runTests(*params):
    
    if len(params) == 2:
        frames = params[0]
        alphas = params[1]
    else:
        frames = [60,65,70,75,80,85,90,95,100]#, 45, 55, 70 ]#10*(n+1) for n in range(100)]
        alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]# , 1.7 ]#1+0.1*n for n in range(11)]

    print("\n ==================================\n Running tests with: \n frames:\t", frames, "\n alpha's:\t ", alphas, "\n")

    trnData = pickle.load(open('trainingData.pkl', 'rb'))
    tstData = pickle.load(open('testingData.pkl', 'rb'))
    xvData  = pickle.load(open('xvData.pkl', 'rb'))
    #print("\n Data loaded \n")
    
    results = {(f,a): [] for f in frames for a in alphas}
    resCur = 0
    result_plots = plt.figure()

    i=1
    for a in alphas:
        for f in frames:
            print( "\n training with: f=%d , a=%f" %(f,a)  )
            model = mod0.em(f,a, trnData)
            print("\n testing...")
            res = evalu.frame_coherency(model, xvData)
            #results[(f,a)] = [model, res]
            print(" coherency: ", res)
            if res > resCur :
                bestMod = model
                resCur = res
                best = (f,a)
            results[(f,a)] = res

    with open('results/2ndFrms_01-13alph.pkl', 'wb') as fl:
        pickle.dump(results, fl, pickle.HIGHEST_PROTOCOL)


def show_results():

    results = get_result_table()
    srtRes = collections.OrderedDict(sorted(results.items()))
    [frames,alphas] = [list(t) for t in zip(*srtRes)]
 
    result_plots = plt.figure()

    for a in alphas:
        y = [results[(f,a)] for f in frames]
        plt.subplot(2,1,1)
        plt.plot(frames,y, label=str('alpha=%f'%a))
        plt.ylabel('coherency')
        plt.xlabel('frames')
        plt.legend(loc=3)

    for f in frames:
        y = [results[(f,a)] for a in alphas]
        plt.subplot(2,1,2)
        plt.plot(alphas,y, label=str('frames=%d'%f))
        plt.ylabel('coherency')
        plt.xlabel('alphas')
        plt.legend(loc=3)

    result_plots.show()


#    print(" Best coherence with: \n frames: ", best[0], "\n alpha: ", best[1] ,"\n coherency on xValidation set: ", results[best][1],
#                  "\n coherency on testset: ", evalu.frame_coherency(model, tstData) )

def get_result_table():

    files = []
    files.append('1stFrms_0203alph')
    files.append('1stFrms_0405alph')
    files.append('1stFrms_067809alph')
    files.append('1stFrms_10111213alph')
    files.append('1stFrms_14-2alph')
    #files.append('2ndFrms_14-2alph')

    results = {}
    for fl in files:
        with open('results/'+ fl + '.pkl', 'rb') as f:
            results.update(pickle.load(f))
    
    return results 
    

pruneData()
splitData()
#runTests()

