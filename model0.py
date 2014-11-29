import numpy as np

# TODO
# - return something useful

args = ('v','s','o')

def em(F, alpha):
    """
    Uses EM to induce frames via the model 0 generative story.
    Arguments:
    F     - the number of frames to induce
    alpha - the dirichlet prior for initializing word distributions (alhpa >= 1)
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
    """

    # Load the data.
    counts = []
    word_to_index = {a: {} for a in args}
    index_to_word = {a: {} for a in args}
    V = {a: 0 for a in args}  
    data = {a: [] for a in args}
    with open("Preprocessing/all_VSOs.sorted.concat") as f:
        for v,s,o,c in map(lambda x: x.split(' ')[:-1], f.read().splitlines()):
            counts.append(int(c))
            for (w, a) in zip((v,s,o), args):
                if not w in word_to_index[a]: 
                    word_to_index[a][w] = V[a]
                    index_to_word[a][V[a]] = w
                    V[a] += 1
                data[a].append(word_to_index[a][w])
    N = len(data)

    # Initialize theta to the uniform distribution.
    theta = np.ones(F) / F
    # Draw from a dirichlet distribution to randomly initialize each frame.
    phi = {a: np.random.dirichlet(np.ones(V[a]) * alpha, F).T for a in args}
    # Create the array for storing posterior estimates.
    mu = np.zeros([N,F])

    t = 0
    while True:
        t += 1 # iteration

        # Check that all our distributions still sum to 1.
        assert(is_prob_dist(theta, .01))
        assert(all(all(is_prob_dist(phi[a][:,f], .01) for f in range(F)) for a in args))

        # E-step
        mu = phi['v'][data['v']] * phi['s'][data['s']] * phi['o'][data['o']] * theta
        print("iteration", t)
        print_clustering(F, mu)

        # M-step
        theta_new = mu.sum(axis=0) / mu.sum()

        phi_new = {}
        w = mu.T * counts
        for (a_i,a) in enumerate(args):
            phi_new[a] = ( np.array([np.bincount(data[a], weights=w[f]) for f in range(F)]).T
                         / np.dot(counts, mu) )

        # Measure how much the distributions have changed from in the previous step.
        delta = (sum(abs(np.subtract(phi[a], phi_new[a])).sum() for a in args) +
                 abs(np.subtract(theta, theta_new)).sum())
        print("delta = ", delta)

        # Relplace the old M-step estimates with the new ones.
        phi = phi_new
        theta = theta_new

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
