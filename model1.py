import random
from probfuncs import normalize, cumulative, cum_dist_choice

def gibbs(F, alpha, beta, T, dataFile):

    data = {} # doc -> list of sentences
    sentence_count = {} # (v,s,o) -> # of observations 
    doc_count = {} # doc (verb) -> # of sentences in it 
    frame_assign = {} # (v,s,o) -> frame assignment 
    frame_count = {} # frame -> # of sentences assigned to it
    frame_count_v = {} # frame -> verb -> # of verb assigned to frame
    frame_count_w = {} # frame -> word -> # of words assigned to frame
    W = set() # number of words (subject + verbs)
    V = set() # number of verbs (documents)

    # read data from file; place in documents
    with open(dataFile) as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            V.add(v)
            W.add(s)
            W.add(o)
            c = int(c)
            if v in data:
                data[v].append((s,o))
                doc_count[v] += c
            else:
                data[v] = [(s,o)]
                doc_count[v] = c
            if (v,s,o) in sentence_count:
                sentence_count[(v,s,o)] += c
            else:
                sentence_count[(v,s,o)] = c

    W_count = len(W)
    V_count = len(V)
    
    # initialize frame assignments and related counts
    frame_count = {f: 0 for f in range(F)}
    frame_count_v = {f: {v: 0 for v in V} for f in range(F)}
    frame_count_w = {f: {w: 0 for w in W} for f in range(F)}
    for v in data:
        frame_count_v[v] = 0
        for (s,o) in data[v]:
            f = random.randrange(F)
            c = sentence_count[(v,s,o)]
            frame_assign[(v,s,o)] = f
            frame_count[f] += c
            frame_count_v[f][v] += c
            frame_count_w[f][s] += c
            frame_count_w[f][o] += c

    # Calculates the probablitiy that (v,s,o) should be assigned frame f
    # given the current counts.
    def posterior(f, v,s,o):
        # how well the verb fits the frame
        v_term  = ( (beta + frame_count_v[f][v]) 
                  / (V_count + frame_count[f]) )
        # how well the subject & object fit the frame
        so_term = ( (2*beta + frame_count_w[f][s] + frame_count_w[f][o]) 
                  / (W_count + frame_count[f]) )
        # how likely the frame is in the document (verb) 
        f_term  = ( (alpha + frame_count_v[f][v])
                  / (F*alpha + sentence_count[(v,s,o)]) )
        return v_term * so_term * f_term

    for t in range(T):
        print('t =',t,end='\r')
        t += 1 # #iteration
        for v in data: # iterate through documents (verbs) 
            for (s,o) in data[v]: # iterate through sentences
                c = sentence_count[(v,s,o)]
                f = frame_assign[(v,s,o)]
                # modify counts to exclude (v,s,o)
                frame_count[f] -= c
                frame_count_v[f][v] -= c
                frame_count_w[f][s] -= c
                frame_count_w[f][o] -= c
                # calculate the posterior for frames
                dist = cumulative(normalize([posterior(f, v,s,o) for f in range(F)]))
                # assign a new label to the sentence
                #f = np.random.choice(range(F), p=dist)
                f = cum_dist_choice(dist)
                # modify counts to reflect (v,s,o)'s new frame
                frame_count[f] += c
                frame_count_v[f][v] += c
                frame_count_w[f][s] += c
                frame_count_w[f][o] += c

    return frame_assign
