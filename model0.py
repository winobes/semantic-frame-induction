import numpy as np
from collections import OrderedDict

args = ('v','s','o')

def em(F, alpha):
    """
    Uses EM to induce frames via the model 0 generative story.
    Arguments:
    F     - the number of frames to induce
    alpha - the dirichlet prior for initializing word distributions (alhpa >= 1)
    Other variables:
    theta - distribution of frames
    """

    # load our tuples into temporary lists 
    # also keep track of all the words it the vocabulary, so we
    # know what the indexes refer to in the end
    # TODO: if we end up using index arrays, some of this conversion
    # to indices should be done in the preprocessing stage.
    vocab = {a: {} for a in args}
    counts = []
    V = {a: 0 for a in args}  # size of the vocabulary
    data = [] 
    with open("Preprocessing/all_VSOs.sorted.concat") as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            counts.append(int(c))
            for (w, a) in zip((v,s,o), args):
                if not w in vocab[a]: 
                    vocab[a][w] = V[a]
                    V[a] += 1
            data.append((vocab['v'][v], vocab['s'][s], vocab['o'][o]))

    # make the index arrays
    N = len(data)

    data = np.array(data).T

    # initialize theta to the uniform distribution
    theta = np.ones(F) / F

    # randomly initialize the argument distributions for each frame
    phi = {a: np.random.dirichlet(np.ones(V[a]) * alpha, F).T for a in args}
    
    mu = np.zeros([N,F])
    t = 0
    while True:
        assert(is_prob_dist(theta, .01))
        assert(all(all(is_prob_dist(phi[a][:,f], .01) for f in range(F)) for a in args))

        t += 1
        # E-step
        mu = phi['v'][data[0]] * phi['s'][data[1]] * phi['o'][data[2]] * theta

        print("iteration", t)
        print_clustering(F, mu)

        # M-step
        theta_new = mu.sum(axis=0) / mu.sum()

        phi_new = {}
        w = mu.T * counts
        for (a_i,a) in enumerate(args):
            phi_new[a] = ( np.array([np.bincount(data[a_i], weights=w[f]) for f in range(F)]).T
                         / np.dot(counts, mu) )

        delta = (sum(abs(np.subtract(phi[a], phi_new[a])).sum() for a in args) +
                 abs(np.subtract(theta, theta_new)).sum())

        phi = phi_new
        theta = theta_new

        print("delta = ", delta)
        if delta < 0.1: 
            return np.argmax(mu, axis=1)

def print_clustering(F, mu):
    frames = np.argmax(mu, axis=1)
    frame_count = {f: 0 for f in range(F)} 
    for f in frames:
        frame_count[f] += 1
    for f in frame_count:
        print(frame_count[f], "unique VSOs from frame", f)

def is_prob_dist(p, epsilon):
    return abs(1 - sum(p)) < epsilon
    
frames = em(10,1.5)
