import model1
import model0
import evaluation
import pickle
import operator


trnData = pickle.load(open('trainingData.pkl', 'rb'))
tstData = pickle.load(open('testingData.pkl', 'rb'))
xvData  = pickle.load(open('xvData.pkl', 'rb'))


def test():
    
    frames = [10,20]#5*(n+1) for n in range(50)]
    alphas = [1] #[0.5,1,1.5,2]
    betas = [0.5,1,1.5,2]

    res_M0 = {}
    res_M1 = {}
    
    for f in frames:
        for a in alphas:
            model = model0.em(f,a,trnData)
            res_M0[(f,a)] = (evaluation.frame_coherency(model,xvData),0)#evaluation.frame_accuracy(model))
            for b in betas:
                model = model1.gibbs(f,a,b,1000, 1, trnData)
                res_M1[(f,a,b)] = (evaluation.frame_coherency(model,xvData),0)#evaluation.frame_accuracy(model))

    metrics = ['coh','acc','both']
    i=0
    for (f,a) in select_best(res_M0):
        fl = open(str('mod0_%s_f%d_a%f.pkl'%(metrics[i],f,a)),'wb')
        pickle.dump(model0.em(f,a,tstData), fl , pickle.HIGHEST_PROTOCOL) 
        i += 1

def select_best(resultsXV):
    
    best_coh = [params for (params,_) in sorted(resultsXV.items(), key=lambda resultsXV: resultsXV[1][0])]
    best_acc = [params for (params,_) in sorted(resultsXV.items(), key=lambda resultsXV: resultsXV[1][1])]

    T = 1000
    for C in range(len(best_coh)):
        A = best_acc.index(best_coh[C])
        cur = (A-C)**2 - (A+C)/2
        if cur < T :
            best_both = best_coh[C]
            T = cur
    return [best_coh[0], best_acc[0], best_both]
    
'''
pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
'''

test()
