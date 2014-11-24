import numpy as np
from collections import OrderedDict

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
    vocab = dict()
    V = 0 # size of the vocabulary
    verbs = []; subjects = []; objects = []
    counts_v = dict(); counts_s = dict(); counts_o = dict();
    with open("Preprocessing/test.concat") as f:
        for v,s,o,c,_ in map(lambda x: x.split(' '), f.read().splitlines()):
            if not v in vocab: vocab[v] = V; V += 1
            if not s in vocab: vocab[s] = V; V += 1
            if not o in vocab: vocab[o] = V; V += 1
            verbs.append(vocab[v])
            subjects.append(vocab[s])
            objects.append(vocab[o])
            c = int(c)
            if vocab[v] in counts_v: counts_v[vocab[v]] += c
            else: counts_v[vocab[v]] = c
            if vocab[s] in counts_s: counts_s[vocab[s]] += c
            else: counts_s[vocab[s]] = c
            if vocab[o] in counts_o: counts_o[vocab[o]] += c
            else: counts_o[vocab[o]] = c

    for w in range(V):
        if not w in counts_v: counts_v[w] = 0
        if not w in counts_s: counts_s[w] = 0
        if not w in counts_o: counts_o[w] = 0

    # make the index arrays
    N = len(verbs)
    verbs = np.array(verbs)
    subjects = np.array(subjects)
    objects = np.array(objects)

    # initialize theta to the uniform distribution
    theta = np.ones(F) / F

    # randomly initialize the argument distributions for each frame
    phi_v = np.random.dirichlet(np.ones(V) * alpha, F).T
    phi_s = np.random.dirichlet(np.ones(V) * alpha, F).T
    phi_o = np.random.dirichlet(np.ones(V) * alpha, F).T

    mu = np.zeros([N,F])
    step = 0
    while True:
        step+=1
        print(step)
        # E-step
        mu = phi_v[verbs] * phi_s[subjects] * phi_o[objects] * theta

        # M-step
        theta_new = mu.sum(axis=0) 
        phi_v_new = np.array([[mu.sum(axis=0)[f] * counts_v[w] for f in range(F)] for w in range(V)])
        phi_s_new = np.array([[mu.sum(axis=0)[f] * counts_s[w] for f in range(F)] for w in range(V)])
        phi_o_new = np.array([[mu.sum(axis=0)[f] * counts_o[w] for f in range(F)] for w in range(V)])

        delta = (abs(np.subtract(phi_v, phi_v_new).sum()) +
                abs(np.subtract(phi_s, phi_s_new).sum()) +
                abs(np.subtract(phi_o, phi_o_new).sum()) +
                abs(np.subtract(theta, theta_new).sum()))

        theta = theta_new
        phi_v = phi_v_new
        phi_s = phi_s_new
        phi_o = phi_o_new

        print(delta)
        if delta < 0.001: 
            return np.argmax(mu, axis=1)

print(em(10, 1.5))
