import model0 as mod0
import evaluation as evalu
import numpy as np
import matplotlib.pyplot as plt
import pickle

args = ('v','s','o')


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
    
    # Load the 
    
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

    results =get_result_table()

"""

with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

            y = [results[(f,a)] for f in frames ]

        
        plt.subplot(a+1,1,i)
        plt.plot(frames,y,marker='o', linestyle='--')
        plt.ylabel('coherency')
        plt.xlabel('frames, with alpha=%f' %a)
        i += 1

    result_plots.tight_layout()
    result_plots.show()
#    print(" Best coherence with: \n frames: ", best[0], "\n alpha: ", best[1] ,"\n coherency on xValidation set: ", results[best][1],
#                  "\n coherency on testset: ", evalu.frame_coherency(model, tstData) )
"""

def get_result_table():

    files = []
    files.append('1stFrms_0203alph')
    files.append('1stFrms_0405alph')
    files.append('1stFrms_067809alph')
    files.append('1stFrms_10111213alph')
    files.append('1stFrms_14-2alph')
    files.append('2ndFrms_14-2alph')

    results = {}
    for f in files:
        with open('results/'+ fl + '.pkl', 'rb') as f:
            results.update(pickle.load(f))
    
    return results 
    

runTests()
