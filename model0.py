import numpy as np

args = ('v','s','o')

def em(F, alpha, inData):
    """
    Uses EM to induce frames via the model 0 generative story.

    Arguments:
    F     - the number of frames to induce
    alpha - the dirichlet prior for initializing word distributions (alpha >= 1)

    Other variables:
    N             - the number of data points (VSO triples).
    V             - for each argument, the size of its vocabulary
    word_to_index - dictionaries (one for each argument) that associates each word in the
                    vocabulary with a unique (integer) index.
    index_to_word - the inverse maps of `word_to_index`.
    data          - three 1XN arrays of data points (as indices).
    counts        - 1XN array of integers. The number of occurances of each VSO in `data`.
    theta         - distribution over F of frames
    phi           - three FXV arrays. For each argument and each frame, a probability
                    distribution over words.
    mu            - an NXF array. For each datapoint, a distribution over frames that
                    gives the (estimated posterior) probability that the frame produced 
                    the observed datapoint

    Returns:
    frame_dists   - a map that for each frame gives a triple that contains, for each argument, a list
                    containing words and their probabilities in the frame, sorted by probability
                    E.g., frame_dists[14]['v'][4] = (8.13e-15, 'like') gives the 5th most common verb
                    ('like') in the 14th frame and tells us that the probablity is 8.13e-15.
    frame_assign  - a map from data points (string triples) to frame indetifiers (integers)
    theta         - (as above)
    """

    # Put the data in the right format for EM
    data = {a: [] for a in args}
    counts = []
    V =  {a: 0 for a in args}
    word_to_index = {a: {} for a in args}
    index_to_word = {a: {} for a in args}
    for ws in inData:
        for (a, w) in zip(args, ws):
            if not w in word_to_index[a]:
                word_to_index[a][w] = V[a]
                index_to_word[a][V[a]] = w
                V[a] += 1
            data[a].append(word_to_index[a][w])
        counts.append(inData[ws])
    data = {a:np.array(data[a]) for a in args}
    N = len(counts)

    # Initialize theta to the uniform distribution.
    theta = np.ones(F) / F
    # Draw from a dirichlet distribution to randomly initialize each frame.
    phi = {a: np.random.dirichlet(np.ones(V[a]) * alpha, F).T for a in args}
    # Create the array for storing posterior estimates.
    mu = np.zeros([N,F])

    t = 0 # iteration
    while True:

        # Check that all our distributions still sum to 1.
        assert(is_prob_dist(theta, .01))
        assert(all(all(is_prob_dist(phi[a][:,f], .01) for f in range(F)) for a in args))

        # E-step
        mu = phi['v'][data['v']] * phi['s'][data['s']] * phi['o'][data['o']] * theta

        # M-step
        theta_new = mu.sum(axis=0) / mu.sum()

        w = mu.T * counts
        phi_new = {a: ( np.array([np.bincount(data[a], weights=w[f]) for f in range(F)]).T
                      / np.dot(counts, mu) ) for a in args}

        # Measure how much the distributions have changed from in the previous step.
        delta = (sum(abs(np.subtract(phi[a], phi_new[a])).sum() for a in args) +
                 abs(np.subtract(theta, theta_new)).sum())

        # Relplace the old M-step estimates with the new ones.
        phi = phi_new
        theta = theta_new

        if delta < 0.1: 

            print('\n')

            frame_dists = {f: {a: {index_to_word[a][i]: prob 
                for (i,prob) in enumerate(phi[a].T[f])} for a in args} for f in range(F)}
            frames = np.argmax(mu, axis=1)
            frame_assign = {tuple([index_to_word[a][data[a][i]] for a in args]): frames[i]
                    for i in range(N)}
            word_data = [tuple(index_to_word[a][data[a][i]] for a in args) for i in range(N)]

            return(frame_dists, frame_assign, theta)

        t += 1 
        print("Iteration:", t, "delta=", delta, end='\r', flush=True)
        #print_clustering(F, mu)

def print_clustering(F, mu):
    frames = np.argmax(mu, axis=1)
    frame_count = {f: 0 for f in range(F)} 
    for f in frames:
        frame_count[f] += 1
    for f in frame_count:
        print(str(frame_count[f]).rjust(8), " unique VSOs in frame", f)
    print("")

def is_prob_dist(p, epsilon):
    return abs(1 - sum(p)) < epsilon
