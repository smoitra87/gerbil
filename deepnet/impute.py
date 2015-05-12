"""Computes partition function for RBM-like models using Annealed Importance Sampling."""
import numpy as np
from deepnet import dbm
from deepnet import util
from deepnet import trainer as tr
from choose_matrix_library import *
import sys
import numpy as np
import pdb
import time
import itertools
import matplotlib.pyplot as plt
from deepnet import visualize
import deepnet
import scipy.io as sio


def LogMeanExp(x):
    offset = x.max()
    return offset + np.log(np.exp(x-offset).mean())

def LogSumExp(x):
    offset = x.max()
    return offset + np.log(np.exp(x-offset).sum())

def Display(w, hid_state, input_state, w_var=None, x_axis=None):
    w = w.asarray().flatten()
    plt.figure(1)
    plt.clf()
    plt.hist(w, 100)
    visualize.display_hidden(hid_state.asarray(), 2, 'activations', prob=True)
    # plt.figure(3)
    # plt.clf()
    # plt.imshow(hid_state.asarray().T, cmap=plt.cm.gray, interpolation='nearest')
    # plt.figure(4)
    # plt.clf()
    # plt.imshow(input_state.asarray().T, cmap=plt.cm.gray, interpolation='nearest')
    #, state.shape[0], state.shape[1], state.shape[0], 3, title='Markov chains')
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.figure(5)
    # plt.clf()
    # plt.suptitle('Variance')
    # plt.plot(np.array(x_axis), np.array(w_var))
    # plt.draw()

def impute_dbm_ais(model):
    """Run approximate pll using AIS on a DBM """


def impute_rbm_gaussian_exact(model):
    """ run exact exact pll and imputation error on an rbm """

    batchsize = model.batchsize
    input_layer = model.GetLayerByName('input_layer') 
    hidden_layer = model.GetLayerByName('bernoulli_hidden1') 
    bern2_hidden_layer = model.GetLayerByName('bernoulli2_hidden1') 
    gaussian_layer = model.GetLayerByName('gaussian_hidden1') 

    # Get input layer features
    dimensions = input_layer.dimensions 
    numlabels = input_layer.numlabels 
    data = input_layer.data

    # set up temp data structures
    for layer in model.layer:
        layer.foo = layer.statesize
        layer.bar = layer.deriv

    zeroslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    onesrow = cm.CUDAMatrix(np.ones([1,\
            batchsize]))
    batchslice = cm.CUDAMatrix(np.zeros([1, batchsize]))
    batchzeroslice = cm.CUDAMatrix(np.zeros([1, batchsize]))
    batchslice2 = cm.CUDAMatrix(np.zeros([1, batchsize]))
    datasize_squared = cm.CUDAMatrix(np.zeros([batchsize, batchsize]))
    datasize_eye = cm.CUDAMatrix(np.eye(batchsize))
    datasize_eye2 = cm.CUDAMatrix(np.eye(batchsize))

    if hidden_layer:
        hidden_bias = hidden_layer.params['bias']
        bedge = next(e for e in model.edge if e.node1.name == 'input_layer' \
                and e.node2.name == 'bernoulli_hidden1')
        w = bedge.params['weight']

    if bern2_hidden_layer:
        bern2_hidden_bias = bern2_hidden_layer.params['bias']
        bedge2 = next(e for e in model.edge if e.node1.name == 'input_layer' \
                and e.node2.name == 'bernoulli2_hidden1')
        w2 = bedge2.params['weight']
    
    if 'bias' in input_layer.params:
        input_bias = input_layer.params['bias']

    if gaussian_layer:
        gedge = next(e for e in model.edge if e.node1.name == 'input_layer' \
                and e.node2.name == 'gaussian_hidden1')
        gw = gedge.params['weight']
        input_diag = input_layer.params['diag']
        diag_val = input_diag.sum() / (input_layer.dimensions  * input_layer.numlabels)




    # RUN Imputation Error
    for dim_idx in range(dimensions):

        #-------------------------------------------
        # Set state of input variables 
        input_layer.GetData()
        dim_offset = dim_idx * numlabels

        for label_idx in range(numlabels):

            batchslice.assign(batchzeroslice)

            #Assign state value
            label_offset = dim_idx * numlabels + label_idx
            input_layer.state.set_row_slice(dim_offset, dim_offset + numlabels, \
                    zeroslice)
            input_layer.state.set_row_slice(label_offset, label_offset+1, onesrow)

            if hidden_layer:
                # Add the contributions from bernoulli hidden layer
                cm.dot(w.T, input_layer.state, target=hidden_layer.state)
                hidden_layer.state.add_col_vec(hidden_bias)
                cm.log_1_plus_exp(hidden_layer.state)
                hidden_layer.state.sum(axis=0, target=batchslice)

            if bern2_hidden_layer:
                # Add the contributions from bernoulli hidden layer
                cm.dot(w2.T, input_layer.state, target=bern2_hidden_layer.state)
                bern2_hidden_layer.state.add_col_vec(bern2_hidden_bias)
                cm.log_1_plus_exp(bern2_hidden_layer.state)
                batchslice.add_sums(bern2_hidden_layer.state, axis=0)

            if 'bias' in input_layer.params:
                cm.dot(input_bias.T, input_layer.state, target=batchslice2)
                batchslice.add_row_vec(batchslice2)

            if gaussian_layer:
                # Add contributions from gaussian hidden layer
                cm.dot(gw.T, input_layer.state, target=gaussian_layer.state)
                cm.dot(gaussian_layer.state.T, gaussian_layer.state, target= datasize_squared)
                datasize_squared.mult(datasize_eye, target=datasize_eye2)
                datasize_eye2.sum(axis=0, target=batchslice2)

                # Add constants from gaussian hidden layer
                integration_constant = gaussian_layer.dimensions * np.log(2*np.pi)
                integration_constant += input_layer.dimensions * diag_val 
                batchslice2.add(integration_constant)
                batchslice2.mult(0.5)
                batchslice.add_row_vec(batchslice2)
            
            input_layer.foo.set_row_slice(label_offset, label_offset+1, batchslice)
 
    # Apply softmax on log Z_v as energies
    input_layer.foo.reshape((numlabels, dimensions * batchsize))        
    input_layer.foo.apply_softmax()

    data.reshape((1, dimensions * batchsize))
    # Calculate Imputation Error
    input_layer.batchsize_temp.reshape((1, dimensions * batchsize))
    input_layer.foo.get_softmax_correct(data, target=input_layer.batchsize_temp)
    input_layer.batchsize_temp.reshape((dimensions, batchsize))
    imperr_cpu = (dimensions - input_layer.batchsize_temp.sum(axis=0).asarray() )/ (0. + dimensions)

    # Calculate Pseudo ll
    input_layer.batchsize_temp.reshape((1, dimensions *  batchsize))
    input_layer.foo.get_softmax_cross_entropy(data, target=input_layer.batchsize_temp, \
            tiny=input_layer.tiny)
    input_layer.batchsize_temp.reshape((dimensions, batchsize))
    pll_cpu = - input_layer.batchsize_temp.sum(axis=0).asarray() 

    # Undo rehapes
    input_layer.foo.reshape((numlabels * dimensions, batchsize))
    data.reshape((dimensions, batchsize))

    zeroslice.free_device_memory()
    onesrow.free_device_memory()
    batchslice.free_device_memory()

    return pll_cpu, imperr_cpu

def impute_rbm_exact(model):
    """ run exact exact pll and imputation error on an rbm """

    batchsize = model.batchsize
    input_layer = model.GetLayerByName('input_layer') 
    hidden_layer = model.GetLayerByName('hidden1') 

    # Get input layer features
    dimensions = input_layer.dimensions 
    numlabels = input_layer.numlabels 
    data = input_layer.data

    # set up temp data structures
    for layer in model.layer:
        layer.foo = layer.statesize
        layer.bar = layer.deriv

    zeroslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    onesrow = cm.CUDAMatrix(np.ones([1,\
            batchsize]))
    batchslice = cm.CUDAMatrix(np.zeros([1, batchsize]))
    batchslice2 = cm.CUDAMatrix(np.zeros([1, batchsize]))

    hidden_bias = hidden_layer.params['bias']
    input_bias = input_layer.params['bias']
    edge = model.edge[0]
    w = edge.params['weight']

    # RUN Imputation Error
    for dim_idx in range(dimensions):

        #-------------------------------------------
        # Set state of input variables 
        input_layer.GetData()
        dim_offset = dim_idx * numlabels

        for label_idx in range(numlabels):
            #Assign state value
            label_offset = dim_idx * numlabels + label_idx
            input_layer.state.set_row_slice(dim_offset, dim_offset + numlabels, \
                    zeroslice)
            input_layer.state.set_row_slice(label_offset, label_offset+1, onesrow)

            cm.dot(w.T, input_layer.state, target=hidden_layer.state)
            hidden_layer.state.add_col_vec(hidden_bias)
            cm.log_1_plus_exp(hidden_layer.state)
            hidden_layer.state.sum(axis=0, target=batchslice)
            cm.dot(input_bias.T, input_layer.state, target=batchslice2)
            batchslice.add_row_vec(batchslice2)

            input_layer.foo.set_row_slice(label_offset, label_offset+1, batchslice)
  
    # Apply softmax on log Z_v as energies
    input_layer.foo.reshape((numlabels, dimensions * batchsize))        
    input_layer.foo.apply_softmax()

    data.reshape((1, dimensions * batchsize))
    # Calculate Imputation Error
    input_layer.batchsize_temp.reshape((1, dimensions * batchsize))
    input_layer.foo.get_softmax_correct(data, target=input_layer.batchsize_temp)
    input_layer.batchsize_temp.reshape((dimensions, batchsize))
    imperr_cpu = (dimensions - input_layer.batchsize_temp.sum(axis=0).asarray() )/ (0. + dimensions)

    # Calculate Pseudo ll
    input_layer.batchsize_temp.reshape((1, dimensions *  batchsize))
    input_layer.foo.get_softmax_cross_entropy(data, target=input_layer.batchsize_temp, \
            tiny=input_layer.tiny)
    input_layer.batchsize_temp.reshape((dimensions, batchsize))
    pll_cpu = - input_layer.batchsize_temp.sum(axis=0).asarray() 

    # Undo rehapes
    input_layer.foo.reshape((numlabels * dimensions, batchsize))
    data.reshape((dimensions, batchsize))

    zeroslice.free_device_memory()
    onesrow.free_device_memory()
    batchslice.free_device_memory()

    return pll_cpu, imperr_cpu

def impute_mf(model, mf_steps, hidden_mf_steps, **opts):
    # Initialize stuff
    batchsize = model.batchsize
    input_layer = model.GetLayerByName('input_layer') 

    hidden_layers = []
    for layer in model.layer:
        if not layer.is_input:
            hidden_layers.append(layer)

    dimensions = input_layer.dimensions 
    numlabels = input_layer.numlabels 
    data = input_layer.data

    # set up temp data structures
    for layer in model.layer:
        layer.foo = layer.statesize

    input_layer.fooslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    input_layer.barslice = cm.CUDAMatrix(np.zeros([1, batchsize]))
    pll = cm.CUDAMatrix(np.zeros([1, batchsize]))
    imputation_err = cm.CUDAMatrix(np.zeros([1, batchsize]))
    
    input_layer.biasslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    input_layer.biasslice.apply_softmax()

    # INITIALIZE TO UNIFORM RANDOM for all layers except clamped layers
    for layer in model.layer:
        layer.state.assign(0)
        layer.ApplyActivation()

    def reshape_softmax(enter=True):
        if enter:
            input_layer.state.reshape((numlabels, dimensions * batchsize))    
            input_layer.foo.reshape((numlabels, dimensions * batchsize))
            data.reshape((1, dimensions * batchsize))
            input_layer.batchsize_temp.reshape((1, dimensions * batchsize))
        else:
            input_layer.state.reshape((numlabels * dimensions, batchsize))
            input_layer.foo.reshape((numlabels * dimensions, batchsize))
            data.reshape((dimensions, batchsize))
            input_layer.batchsize_temp.reshape((dimensions, batchsize))

    # RUN Imputation Error
    for dim_idx in range(dimensions):

        #-------------------------------------------
        # Set state of input variables
        input_layer.GetData()
        offset = dim_idx * numlabels
        input_layer.state.set_row_slice(offset, offset + numlabels, \
                input_layer.biasslice)

        for layer in model.layer:
            if not layer.is_input:
                layer.state.assign(0)

        # Run MF steps
        for mf_idx in range(mf_steps):
            for hid_mf_idx in range(hidden_mf_steps):
                for layer in hidden_layers:
                    model.ComputeUp(layer, train=False, compute_input=False, step=0,
                        maxsteps=0, use_samples=False, neg_phase=False)
            model.ComputeUp(input_layer, train=False, compute_input=True, step=0,
                    maxsteps=0, use_samples=False, neg_phase=False)

            input_layer.state.get_row_slice(offset, offset + numlabels , \
                target=input_layer.fooslice)

            input_layer.GetData()
            input_layer.state.set_row_slice(offset, offset + numlabels , \
                input_layer.fooslice)

        # Calculate pll
        reshape_softmax(enter=True)
        input_layer.state.get_softmax_cross_entropy(data,\
                target=input_layer.batchsize_temp, tiny=input_layer.tiny)
        reshape_softmax(enter=False)

        input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                target=input_layer.barslice)
        pll.add_sums(input_layer.barslice, axis=0)
        
        # Calculate imputation error
        if 'blosum90' in opts:
            reshape_softmax(enter=True)
            input_layer.state.get_softmax_blosum90(data, target=input_layer.batchsize_temp)
            reshape_softmax(enter=False)

            input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                    target=input_layer.barslice)
            imputation_err.add_sums(input_layer.barslice, axis=0)
        else:
            reshape_softmax(enter=True)
            input_layer.state.get_softmax_correct(data, target=input_layer.batchsize_temp)
            reshape_softmax(enter=False)

            input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                    target=input_layer.barslice)
            imputation_err.add_sums(input_layer.barslice, axis=0, mult=-1.)
            imputation_err.add(1.)

    #--------------------------------------
    # free device memory for newly created arrays
    pll_cpu = -pll.asarray()
    imperr_cpu = imputation_err.asarray()
    imperr_cpu /= (dimensions+0.)

    input_layer.fooslice.free_device_memory()
    input_layer.biasslice.free_device_memory()
    input_layer.barslice.free_device_memory()
    pll.free_device_memory()
    imputation_err.free_device_memory()

    return pll_cpu, imperr_cpu

def multicol_mf(model, multicols, **opts):
    # Initialize stuff
    batchsize = model.batchsize
    input_layer = model.GetLayerByName('input_layer') 

    hidden_layers = []
    for layer in model.layer:
        if not layer.is_input:
            hidden_layers.append(layer)

    dimensions = input_layer.dimensions 
    numlabels = input_layer.numlabels 
    data = input_layer.data

    # set up temp data structures
    for layer in model.layer:
        layer.foo = layer.statesize

    input_layer.fooslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    input_layer.barslice = cm.CUDAMatrix(np.zeros([1, batchsize]))
    pll = cm.CUDAMatrix(np.zeros([1, batchsize]))
    imputation_err = cm.CUDAMatrix(np.zeros([1, batchsize]))
    
    input_layer.biasslice = cm.CUDAMatrix(np.zeros([input_layer.numlabels,\
            batchsize]))
    input_layer.biasslice.apply_softmax()

    # Get the multicol dimensions
    nBlocks, nCols = multicols.shape

    # INITIALIZE TO UNIFORM RANDOM for all layers except clamped layers
    for layer in model.layer:
        layer.state.assign(0)
        layer.ApplyActivation()

    def reshape_softmax(enter=True):
        if enter:
            input_layer.state.reshape((numlabels, dimensions * batchsize))    
            input_layer.foo.reshape((numlabels, dimensions * batchsize))
            data.reshape((1, dimensions * batchsize))
            input_layer.batchsize_temp.reshape((1, dimensions * batchsize))
        else:
            input_layer.state.reshape((numlabels * dimensions, batchsize))
            input_layer.foo.reshape((numlabels * dimensions, batchsize))
            data.reshape((dimensions, batchsize))
            input_layer.batchsize_temp.reshape((dimensions, batchsize))


    # RUN Imputation Error
    for mult_idx in range(nBlocks):
        #-------------------------------------------
        # Set state of input variables
        input_layer.GetData()
        for col_idx in range(nCols):
            dim_idx = multicols[mult_idx, col_idx]
            offset = dim_idx * numlabels
            input_layer.state.set_row_slice(offset, offset + numlabels, \
                    input_layer.biasslice)

        for layer in model.layer:
            if not layer.is_input:
                layer.state.assign(0)

        for layer in hidden_layers:
            model.ComputeUp(layer, train=False, compute_input=False, step=0,
                        maxsteps=0, use_samples=False, neg_phase=False)
        model.ComputeUp(input_layer, train=False, compute_input=True, step=0,
                    maxsteps=0, use_samples=False, neg_phase=False)

        # Calculate pll
        reshape_softmax(enter=True)
        input_layer.state.get_softmax_cross_entropy(data,\
                target=input_layer.batchsize_temp, tiny=input_layer.tiny)
        reshape_softmax(enter=False)

        for col_idx in range(nCols):
            dim_idx = multicols[mult_idx, col_idx]
            input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                    target=input_layer.barslice)
            pll.add_sums(input_layer.barslice, axis=0)
        
        # Calculate imputation error
        if 'blosum90' in opts:
            reshape_softmax(enter=True)
            input_layer.state.get_softmax_blosum90(data, target=input_layer.batchsize_temp)
            reshape_softmax(enter=False)

            for col_idx in range(nCols):
                dim_idx = multicols[mult_idx, col_idx]
                input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                        target=input_layer.barslice)
                imputation_err.add_sums(input_layer.barslice, axis=0)
        else:
            reshape_softmax(enter=True)
            input_layer.state.get_softmax_correct(data, target=input_layer.batchsize_temp)
            reshape_softmax(enter=False)

            for col_idx in range(nCols):
                dim_idx = multicols[mult_idx, col_idx]
                input_layer.batchsize_temp.get_row_slice(dim_idx, dim_idx + 1 , \
                        target=input_layer.barslice)
                imputation_err.add_sums(input_layer.barslice, axis=0, mult=-1.)
                imputation_err.add(1.)

    #--------------------------------------
    # free device memory for newly created arrays
    pll_cpu = -pll.asarray()
    imperr_cpu = imputation_err.asarray()
    imperr_cpu /= (nBlocks * nCols +0.)

    input_layer.fooslice.free_device_memory()
    input_layer.biasslice.free_device_memory()
    input_layer.barslice.free_device_memory()
    pll.free_device_memory()
    imputation_err.free_device_memory()

    return pll_cpu, imperr_cpu


def Usage():
    print '%s <model file> <number of Markov chains to run> [number of words (for Replicated Softmax models)]'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run AIS")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--infer-method", type=str, default='exact', \
            help='mf/gibbs/exact/gaussian_exact')
    parser.add_argument("--mf-steps", type=int, default=1)
    parser.add_argument("--hidden-mf-steps", type=int, default=1)
    parser.add_argument("--outf", type=str, help='Output File')
    parser.add_argument("--valid_only", action='store_true', help="only run the validation set")
    parser.add_argument("--blosum90", action='store_true', help="Calculate blosum90 scores")
    parser.add_argument("--ncols", type=int, help="Number of multiple columns")
    parser.add_argument("--multmode", type=str, help="Multicol mode",default='rand')
    args = parser.parse_args()

    if not args.outf : 
        raise ValueError('Output file not defined')

    if not args.train_file or not args.model_file : 
        raise ValueError('Models and data missing')

    board = tr.LockGPU()
    model_file = args.model_file
    train_file = args.train_file
    model = dbm.DBM(model_file, train_file)

    trainer_pb = util.ReadOperation(train_file)
    dataset = os.path.basename(trainer_pb.data_proto_prefix)

    # Fix paths
    dirname = os.path.split(model.t_op.data_proto_prefix)[1]
    import awsutil
    deepnet_path = awsutil.get_deepnet_path()
    model.t_op.data_proto_prefix = os.path.join(deepnet_path, 'datasets/',\
            dirname)
    model.t_op.skip_last_piece = False
    model.t_op.get_last_piece = True
    model.t_op.randomize = False

    model.LoadModelOnGPU()
    model.SetUpData()
    
    if args.valid_only:
        data_types = ['valid']
    else:
        data_types = ['train', 'valid', 'test']

    datagetters = {
            'train' : model.GetTrainBatch,
            'valid' : model.GetValidationBatch,
            'test' : model.GetTestBatch
            }
    batchsizes = {
            'train' : model.train_data_handler.num_batches,
            'valid' : model.validation_data_handler.num_batches,
            'test' : model.test_data_handler.num_batches
            }

    opts = {}

    cm.CUDAMatrix.init_random(seed=int(time.time()))

    if len(model.layer) > 2 and args.infer_method=='exact':
        raise ValueError('Cannot use exact Exact inference for DBMs')

    from collections import defaultdict    
    pll_data = defaultdict(list)
    imperr_data = defaultdict(list)
    for data_type in data_types:
        num_batches = batchsizes[data_type]
        datagetter = datagetters[data_type]
        for batch_idx in range(num_batches):
            print("Evalutating batch {}".format(batch_idx+1))
            datagetter()

            if args.infer_method == 'mf':
                if args.blosum90:
                    pll, imperr = impute_mf(model, args.mf_steps, args.hidden_mf_steps, blosum90=True)
                else:
                    pll, imperr = impute_mf(model, args.mf_steps, args.hidden_mf_steps)
            elif args.infer_method == 'multicol':
                ncols = args.ncols;
                multicol_file = 'datasets/{0}/multicol/{1}_{2}.mat'.format(dataset,args.multmode, ncols)
                multicols = sio.loadmat(multicol_file)['multicols']
                multicols = np.asarray(multicols, dtype=np.int)
                multicols = multicols - 1; # convert from matlab indexing
                if args.blosum90:
                    pll, imperr = multicol_mf(model, multicols, blosum90=True)
                else:
                    pll, imperr = multicol_mf(model, multicols)
            elif args.infer_method == 'exact':
                pll, imperr = impute_rbm_exact(model)
            elif args.infer_method == 'gaussian_exact':
                pll, imperr = impute_rbm_gaussian_exact(model)
            else:
                raise ValueError("Unknown infer method")

            pll, imperr = pll.flatten(), imperr.flatten()
            pll_data[data_type].append(pll)
            imperr_data[data_type].append(imperr)

        pll_data[data_type] = np.concatenate(pll_data[data_type])
        imperr_data[data_type] = np.concatenate(imperr_data[data_type])

    #-------------------------------------------------------------------
    # Print and save the results 
    for dtype in pll_data :
        pll = pll_data[dtype]
        imperr = imperr_data[dtype]
        print '%s : Pseudo-LogLikelihood %.5f, std %.5f' % (dtype, pll.mean(), pll.std())
        print '%s : Imputation Error %.5f, std %.5f' % (dtype, imperr.mean(), imperr.std())

    tr.FreeGPU(board)

    import pickle
    with open(args.outf,'wb') as fout:
        pkldata = { 'pll' : pll_data, 'imperr' : imperr_data }
        pickle.dump(pkldata, fout)

