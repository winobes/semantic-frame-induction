import model0 as mod0
import evaluation as evalu
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import os
from probfuncs import weighted_sample

args = ('v','s','o')

def pruneData( dataFile='Preprocessing/all_VSOs.sorted.concat', cutoffPC=20):
    print('Pruning to %', cutoffPC, '... ', end='')
    V = {}
    data = {}
    with open(dataFile) as f:
        # build the verb vocabulary
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            c = int(c)
            if v in V:
                V[v] += c
            else: 
                V[v] = c
            if (v,s,o) in data:
                data[(v,s,o)] += c
            else:
                data[(v,s,o)] = c

    # determine the cutoff 
    counts = list(V.values())
    counts.sort(reverse=True)
    cutoff = counts[int(len(counts) * cutoffPC/100)]
    print("The absolute cutoff for verb frequency will be", cutoff)
    # pop off the data with low counts
    for (v,s,o) in data.copy():
        if V[v] < cutoff:
            data.pop((v,s,o))

    pickle.dump(data, open("allData.pkl", 'wb'))
    print('Done.')

def splitData( dataFile='allData.pkl', testPC=10, xvPC=20):

    # load up all the data
    if not os.path.isfile(dataFile):
        raise ValueError ("must run pruneData first.")
    data = pickle.load(open(dataFile, 'rb'))
    trainingData = data.copy()

    # make a weighted sample of data
    dataList = [s for s in data]
    weights = [data[s] for s in dataList]

    # genenic function for generating the samples from w/o replacement
    def sample_from_data(PC): 
        size = int(len(data) * PC/100) 
        sampleList = weighted_sample(dataList, weights, size)

        # put the sample in a good format and adjust the data counts
        sampleData = {}
        while sampleList:
            vso = sampleList.pop()
            trainingData[vso] -= 1
            if vso in sampleData: 
                sampleData[vso] += 1
                if trainingData[vso] == 0:
                    trainingData.pop(vso)
            else: 
                sampleData[vso] = 1
        return sampleData

    # generate both samples
    print('Splitting off %', testPC, ' for training data...')
    testData = sample_from_data(testPC)
    print('Splitting off %', xvPC, ' for cross verb data...')
    xvData   = sample_from_data(xvPC)

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
    pickle.dump(data, open("trainingData.pkl", 'wb'))
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
    

#pruneData()
#splitData()
#runTests()

