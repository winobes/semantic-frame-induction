
import argparse
import model0

argparser = argparse.ArgumentParser()
argparser.add_argument("-f","-F","--frames", help="number of frames to induce", type=int, required=True)
argparser.add_argument("-a","--alpha", help="dirichlet prior for initializing word distributions", type=float, required=True)
args = argparser.parse_args()

model0.em(args.frames, args.alpha)
