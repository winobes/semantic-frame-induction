import model1
import model0
import evaluation
import pickle
import operator
import multiprocessing
import itertools


trnData = pickle.load(open('trainingData.pkl', 'rb'))
tstData = pickle.load(open('testingData.pkl', 'rb'))
xvData  = pickle.load(open('xvData.pkl', 'rb'))

def train0( prmS ):
    (f,a) = prmS
    model = model0.em(f,a,trnData)
    return ((f,a),(evaluation.frame_coherency(model,xvData),evaluation.frame_accuracy(verb_dists(model), .01, 5 )))

def train1( prmS ):
    (f,a,b) = prmS
    model = model1.gibbs(f,a,b,2, 1, trnData)
    return ((f,a,b), (evaluation.frame_coherency(model,xvData),evaluation.frame_accuracy(model[0], .01, 5)))
    
def test():
    
    frames = [25,50]#,100,150,200]
    alphas = [.1]#,.5,1,1.5,2]
    betas = [.1]#,.5,1,1.5,2]

    params0 = itertools.product(frames,alphas)
    params1 = itertools.product(frames,alphas,betas)

    pool = multiprocessing.Pool(4)
    res_M0 = dict(pool.map(train0, params0 ))
    res_M1 = dict(pool.map(train1, params1 ))
    
    metrics = ['coh','acc','both']
    i=0
    for (f,a) in select_best(res_M0):
        print(" M0: best %s with %d frames - alpha: %f"%(metrics[i],f,a))
        #fl = open(str('mod0_%s_f%d_a%f.pkl'%(metrics[i],f,a)),'wb')
        #pickle.dump(model0.em(f,a,tstData), fl , pickle.HIGHEST_PROTOCOL) 
        i += 1

    i=0
    for f,a,b in select_best(res_M1):
        print(" M1: best %s with %d frames - alpha: %f - beta: %f"%(metrics[i],f,a,b))
        #fl = open(str('mod1_%s_f%d_a%f_b%f.pkl'%(metrics[i],f,a,b)),'wb')
        #pickle.dump(model1.gibbs(f,a,b,10,1,tstData), fl , pickle.HIGHEST_PROTOCOL) 
        i += 1

def select_best(resultsXV):
    
    best_coh = [params for (params,_) in sorted(resultsXV.items(), key=lambda resultsXV: resultsXV[1][0])]
    best_acc = [params for (params,_) in sorted(resultsXV.items(), key=lambda resultsXV: resultsXV[1][1])]


    # best parameters for both:
    #     euclidian distance in sorted arrays of best coherency/accuracy - average distance to best params of individual metrics
    T = 1000
    for C in range(len(best_coh)):
        A = best_acc.index(best_coh[C])
        cur = (A-C)**2 - (A+C)/2
        if cur < T :
            best_both = best_coh[C]
            T = cur
    return [best_coh[len(best_coh)-1], best_acc[len(best_coh)-1], best_both]

def verb_dists(model):
    dists = model[0]
    verb_dists = {frm: dists[frm]['v'] for frm in dists}

    return verb_dists

#test()
