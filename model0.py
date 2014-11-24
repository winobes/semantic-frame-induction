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
    vocab = {'v': {}, 's': {}, 'o': {}}
    counts = {'v': {}, 's': {}, 'o': {}}
    V = {'v': 0, 's': 0, 'o': 0}  # size of the vocabulary
    data = [] 
    with open("Preprocessing/test.concat") as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            for (w, a) in zip((v,s,o), ('v','s','o')):
                if not w in vocab[a]: 
                    vocab[a][w] = V[a]
                    V[a] += 1
                if vocab[a][w] in counts[a]: 
                    counts[a][vocab[a][w]] += int(c)
                else:
                    counts[a][vocab[a][w]] = int(c)
            data.append((vocab['v'][v], vocab['s'][s], vocab['o'][o]))

    # make the index arrays
    N = len(data)

    data = np.array(data).T

    # initialize theta to the uniform distribution
    theta = np.ones(F) / F

    # randomly initialize the argument distributions for each frame
    phi_v = np.random.dirichlet(np.ones(V['v']) * alpha, F).T
    phi_s = np.random.dirichlet(np.ones(V['s']) * alpha, F).T
    phi_o = np.random.dirichlet(np.ones(V['o']) * alpha, F).T

    mu = np.zeros([N,F])
    while True:
        # E-step
        mu = phi_v[data[0]] * phi_s[data[1]] * phi_o[data[2]] * theta

        # M-step
        theta_new = mu.sum(axis=0) 
        phi_v_new = np.array([[mu.sum(axis=0)[f] * counts['v'][w] for f in range(F)] for w in range(V['v'])])
        phi_s_new = np.array([[mu.sum(axis=0)[f] * counts['s'][w] for f in range(F)] for w in range(V['s'])])
        phi_o_new = np.array([[mu.sum(axis=0)[f] * counts['o'][w] for f in range(F)] for w in range(V['o'])])

        delta = (abs(np.subtract(phi_v, phi_v_new).sum()) +
                abs(np.subtract(phi_s, phi_s_new).sum()) +
                abs(np.subtract(phi_o, phi_o_new).sum()) +
                abs(np.subtract(theta, theta_new).sum()))

        theta = theta_new
        phi_v = phi_v_new
        phi_s = phi_s_new
        phi_o = phi_o_new

        print("delta =", delta)
        if delta < 0.001: 
            return np.argmax(mu, axis=1)

print(em(10,1.5))
