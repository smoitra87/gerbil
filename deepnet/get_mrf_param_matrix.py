"""Write a model protocol buffer to mat file."""
from deepnet import util
import numpy as np
import sys
import scipy.io
import scipy.io as sio
import gzip
import os


def Convert(mat_file, out_file):
    """ Create the necesarry things"""
    matfile = sio.loadmat(mat_file)

    # get the weight matrix
    weight = np.asarray(matfile['mrf_weights'], dtype='float32')
    
    np.save(out_file, weight)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mat_file", type=str)
    parser.add_argument("--out_file", type=str)

    args = parser.parse_args()
    
    Convert(args.mat_file, args.out_file)


