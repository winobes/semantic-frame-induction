import model0 as mod0
import evaluation as evalu
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import os
from probfuncs import weighted_sample

args = ('v','s','o')

def pruneData( dataFile='Preprocessing/all_VSOs.sorted.concat', verbPC=20):
    print('Pruning to %', verbPC, '... ', end='')
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
    cutoff = counts[int(len(counts) * verbPC/100)]
    print("The cutoff for verb frequency will be", cutoff)
    # pop off the data with low counts
    for (v,s,o) in data.copy():
        if V[v] < cutoff:
            data.pop((v,s,o))

    pickle.dump(data, open("allData.pkl", 'wb'))
    print('Done.')

def splitData( dataFile='allData.pkl', testPC=10):

    print('Splitting off %', testPC, '...')

    # load up all the data
    if not os.path.isfile(dataFile):
        raise ValueError ("must run pruneData first.")
    data = pickle.load(open(dataFile, 'rb'))

    # make a weighted sample of data
    dataList = [s for s in data]
    weights = [data[s] for s in dataList]
    testSize = int(len(data) * testPC/100) 
    testDataList = weighted_sample(dataList, weights, testSize)

    # put the sample in a good format and adjust the training data counts
    testData = {}
    while testDataList:
        vso = testDataList.pop()
        if vso in testData:
            testData[vso] += 1
            data[vso] -= 1
            if data[vso] == 0:
                data.pop(vso)
        else: 
            testData[vso] = 1

    pickle.dump(data, open("trainingData.pkl", 'wb'))
    pickle.dump(testData, open("testingData.pkl", 'wb'))


def loadData( dataFile , trainPC, xvalidPC, testPC):

    counts = []
    word_to_index = {a: {} for a in args}
    index_to_word = {a: {} for a in args}
    V = {a: 0 for a in args}  
    trnData = {a: [] for a in args}

    xvData = []
    tstData= []

    nD = 0
    nTrn = 0
    nXval = 0
    nTst = 0
    

    
    with open( dataFile) as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            c = int(c)
            if c > 10:
                nD += 1 
                splt = c*(trainPC/100)
                counts.append(int(splt))
                for (w, a) in zip((v,s,o), args):
                    if not w in word_to_index[a]: 
                        word_to_index[a][w] = V[a]
                        index_to_word[a][V[a]] = w
                        V[a] += 1
                    trnData[a].append(word_to_index[a][w])
                splt2 = int((c-splt)* (xvalidPC/100)) 
                xvData.append((v,s,o, splt2))
                tstData.append((v,s,o, c-splt-splt2))

    #print("\n Data lengths: train, xval, test: ", len(trnData['v']), len(xvData), len(tstData))
                                        
    return ((counts, word_to_index, index_to_word, V, trnData), xvData, tstData)
    

def runTests(*params):
    
    if len(params) == 2:
        frames = params[0]
        alphas = params[1]
    else:
        frames = [60,65,70,75,80,85,90,95,100]#, 45, 55, 70 ]#10*(n+1) for n in range(100)]
        alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]# , 1.7 ]#1+0.1*n for n in range(11)]

    print("\n ==================================\n Running tests with: \n frames:\t", frames, "\n alpha's:\t ", alphas, "\n")

    trnData, xvData, tstData = loadData("all_VSOs.sorted.concat",70,20,10)
    #print("\n Data loaded \n")
    
    results = {(f,a): [] for f in frames for a in alphas}
    resCur = 0
    result_plots = plt.figure()

    i=1
    for a in alphas:
        for f in frames:
            print( "\n training with: f=%d , a=%f" %(f,a)  )
            model = mod0.em(f,a,*trnData)
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
    

#runTests()
pruneData()
splitData()
