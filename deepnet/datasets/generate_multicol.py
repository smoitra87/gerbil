import os, sys
from argparse import ArgumentParser
import numpy as np
import random
from operator import mul
from fractions import Fraction
import itertools
import scipy.io as sio
import pickle

def nCk(n,k):
    return int( reduce(mul, (Fraction(n-i,i+1) for i in range(k)), 1))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pfamid", type=str)
    parser.add_argument("--mode", type=str, help="rand/other")
    parser.add_argument("--nCols", type=int,nargs="+", help="2 3 5 10")
    parser.add_argument("--nBlocks", type=int, help="Num blocks", default=100)

    args = parser.parse_args()

    X = np.load('{0}/{0}_train.npy'.format(args.pfamid))
    nRes = X.shape[1]

    random.seed(42)
    outf_str = '{0}/multicol/{1}_{2}.mat'
    if args.mode == 'rand':
        for nCol in args.nCols:
            if args.nBlocks > nCk(nRes, nCol):
                multicols = list(itertools.combinations(range(1,nRes+1),nCol))
            else:
                multicols = [sorted(random.sample(range(1,nRes+1), nCol)) for _ in xrange(args.nBlocks)]
            outf = outf_str.format(args.pfamid,args.mode,nCol)
            sio.savemat(outf, {'multicols': np.asarray(multicols)})
    elif args.mode == 'block':
        for nCol in args.nCols:
            multicols = [range(i,i+nCol) for i in range(1, nRes - nCol + 2)]
            if len(multicols) > args.nBlocks:
                multicols = random.sample(multicols, args.nBlocks)
            outf = outf_str.format(args.pfamid,args.mode,nCol)
            sio.savemat(outf, {'multicols': np.asarray(multicols)})
    elif args.mode == 'core' or args.mode == 'surface':
        with open(os.path.join(args.pfamid, "functional.pkl"),'rb') as fin:
            functional = pickle.load(fin)

        res = functional['msa-{}'.format(args.mode)]
        nRes = len(res)
        for nCol in args.nCols:
            if args.nBlocks > nCk(nRes, nCol):
                multicols = list(itertools.combinations(res, nCol))
            else:
                multicols = [sorted(random.sample(res, nCol)) for _ in xrange(args.nBlocks)]
            outf = outf_str.format(args.pfamid,args.mode,nCol)
            print outf, multicols
            sio.savemat(outf, {'multicols': np.asarray(multicols)})
    elif args.mode == 'bound':
        with open(os.path.join(args.pfamid, "functional.pkl"),'rb') as fin:
            functional = pickle.load(fin)

        res = functional['msa-interface'] + functional['msa-ligands']
        res = sorted(list(set(res)))
        nRes = len(res)
        for nCol in args.nCols:
            if args.nBlocks > nCk(nRes, nCol):
                multicols = list(itertools.combinations(res, nCol))
            else:
                multicols = [sorted(random.sample(res, nCol)) for _ in xrange(args.nBlocks)]
            outf = outf_str.format(args.pfamid,args.mode,nCol)
            print outf, multicols
            sio.savemat(outf, {'multicols': np.asarray(multicols)})
    else:
        raise ValueError('Unknown mode')
