"""Write a model protocol buffer to mat file."""
from deepnet import util
import numpy as np
import sys
import scipy.io
import scipy.io as sio
import gzip
import os


def Convert(args, dirpath, mat_file,  dump_npy = False, out_file = 'rbm_mrf', model_file=None):
    """ Create the necesarry things"""
    matfile = sio.loadmat(mat_file)

    if args.minfill:
        # get the weight matrix
        weight = np.asarray(matfile['minL'], dtype='float32')
        Pmat = matfile['Pmat']
        weight = weight.dot(Pmat)
        weight = weight.T
    elif args.random:
        # get the weight matrix
        weight = np.asarray(matfile['L'].T, dtype='float32')
        weight[np.abs(weight)<1e-8] = 0.0
        nnz = np.count_nonzero(weight)
        nNodes = weight.shape[0]
        weight = np.random.randn(nNodes**2, 1)
        rangeIdx = np.arange(nNodes**2)
        np.random.shuffle(rangeIdx)
        weight[rangeIdx[nnz:]] = 0.
        weight = weight.reshape(nNodes, nNodes)
        print("nnz: {}".format(nnz))
    elif args.thresh:
        # get both the weight materices
        if not -0.00001 < args.thresh < 100.000001:
            raise ValueError("Threshold should be b/w 0 and 1")

        weight = np.asarray(matfile['L'].T, dtype='float32')
        weight_minfill = np.asarray(matfile['minL'], dtype='float32')
        nnz = np.sum(np.abs(weight) > 1e-10)
        nnz_minfill = np.sum(np.abs(weight_minfill) > 1e-10)
        num_to_delete = int((nnz - nnz_minfill) * args.thresh / 100.)
        weight_nnz = np.abs(weight[np.abs(weight)>1e-10])
        threshold = np.sort(weight_nnz)[num_to_delete]
        weight[np.abs(weight) < threshold] = 0.0
    else:
        weight = np.asarray(matfile['L'].T, dtype='float32')

    nFeats,_ = weight.shape
    diag = np.ones([nFeats, 1]) * matfile['min_eig'] * (1+matfile['alpha'])
    diag = np.asarray(diag, dtype='float32')
    
    if dump_npy : 
        if args.edge_input_file:
            edge_file = os.path.join(dirpath, args.edge_input_file)
        else:
            edge_file = os.path.join(dirpath, 'edge_input_to_gaussian.npy')
        diag_file = os.path.join(dirpath, 'diag_gaussian.npy')
        np.save(edge_file, weight)
        np.save(diag_file, diag)
    else:
        model = util.ReadModel(model_file)
        proto_weight = next(param for param in model.edge[0].param if param.name == 'weight')
        proto_weight.mat = util.NumpyAsParameter(weight)
        proto_weight.dimensions.extend(weight.shape)

        input_layer = next(l for l in model.layer if l.name == 'input_layer')
        proto_diag = next(param for param in input_layer.param if param.name == 'diag')
        proto_diag.mat = util.NumpyAsParameter(diag)
        proto_diag.dimensions.extend(diag.shape)

        out_file = os.path.join(dirpath, out_file)
        f = gzip.open(out_file, 'wb')
        f.write(model.SerializeToString())
        f.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--mat_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--edge_input_file", type=str)
    parser.add_argument("--dirpath", type=str)
    parser.add_argument("--npy", action='store_true')
    parser.add_argument("--minfill", action='store_true')
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--thresh", type=float, help="Threshold L")

    args = parser.parse_args()
    
    if args.npy:
        Convert(args, args.dirpath, args.mat_file, dump_npy = True)
    else:
        Convert(args, args.dirpath, args.mat_file, dump_npy = False, out_file = args.out_file, model_file=args.model_file)


