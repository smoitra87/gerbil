import os, sys
from argparse import ArgumentParser
import numpy as np


if __name__ == '__main__':
    fname = sys.argv[1]
    outputf = sys.argv[2]
    X = np.load(fname)
    X = np.asarray(abs(X)>1e-8, dtype='float32')
    np.save(outputf, X)
