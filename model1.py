import random
from probfuncs import normalize, cumulative, cum_dist_choice

def gibbs(F, alpha, beta, T, inData):

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

    # infer the frame prior from assignments
    theta = normalize([frame_count[f] for f in range(F)])
    # infer frame's argument distributons from assignments
    frame_dists = construct_frame_dists(frame_assign, F, sentence_count)

    return (frame_dists, frame_assign, theta)

# questionable.
def construct_frame_dists(frame_assign, F, counts):
    totals_in_frame = {f: {a: 0 for a in args} for f in range(F)}
    frame_dists = {f: {a: {} for a in args} for f in range(F)}
    for ((v,s,o), f) in frame_assign.items():
        for (a,w) in zip(args, (v,s,o)):
            if w in frame_dists[f][a]:
                frame_dists[f][a][w] += counts[(v,s,o)]
            else:
                frame_dists[f][a][w] = counts[(v,s,o)]
