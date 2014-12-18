import random
from probfuncs import normalize, cumulative, cum_dist_choice

args = ('v', 's', 'o')

def gibbs(F, alpha, beta, T, burnIn, inData):

    data = {} # doc -> list of sentences
    sentence_count = inData # (v,s,o) -> # of observations 
    doc_count = {} # doc (verb) -> # of sentences in it 
    frame_assign = {} # (v,s,o) -> frame assignment 
    frame_count = {} # frame -> # of sentences assigned to it
    frame_count_v = {} # frame -> verb -> # of verb assigned to frame
    frame_count_w = {} # frame -> word -> # of subjects/verbs assigned to frame
    W = set() # number of words (subject + verbs)
    V = set() # number of verbs (documents)

    for ((v,s,o),c) in sentence_count.items():
        V.add(v)
        W.add(s)
        W.add(o)
        if v in data:
            data[v].append((s,o))
            doc_count[v] += c
        else:
            data[v] = [(s,o)]
            doc_count[v] = c

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

    samples = {(v,s,o): {f: 0 for f in range(F)} for (v,s,o) in sentence_count}
    
    for t in range(T):
        print('t =',t,'of',T,end='\r')
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
                f = cum_dist_choice(dist)
                frame_assign[(v,s,o)] = f
                if t > burnIn:
                    samples[(v,s,o)][f] += 1
                # modify counts to reflect (v,s,o)'s new frame
                frame_count[f] += c
                frame_count_v[f][v] += c
                frame_count_w[f][s] += c
                frame_count_w[f][o] += c

    return samples
