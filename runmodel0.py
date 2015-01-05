
import pickle
import argparse
import model0

argparser = argparse.ArgumentParser()
argparser.add_argument("-f","-F","--frames", help="number of frames to induce", type=int, required=True)
argparser.add_argument("-a","--alpha", help="dirichlet prior for initializing word distributions", type=float, required=True)
args = argparser.parse_args()
frames = args.frames
alpha = args.alpha

data = pickle.load(open('trainingData.pkl', 'rb'))
filename = "sample_m0_F"+str(frames)+"_alpha"+str(alpha)+".pkl"

result = model0.em(frames, alpha, data)
pickle.dump(result, open(filename, 'wb'))
