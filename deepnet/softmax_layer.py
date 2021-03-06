from layer import *

class SoftmaxLayer(Layer):
  def __init__(self, *args, **kwargs):
    super(SoftmaxLayer, self).__init__(*args, **kwargs)

  @classmethod
  def IsLayerType(cls, proto):
    return proto.hyperparams.activation == deepnet_pb2.Hyperparams.SOFTMAX

  def ApplyActivation(self):
    # Reshape the state to perform softmax
    dimensions = self.dimensions
    batchsize = self.batchsize
    numlabels = self.numlabels
    self.state.reshape((numlabels, dimensions * batchsize))

    self.state.apply_softmax()

    # Undo Reshape
    self.state.reshape((numlabels * dimensions, batchsize))

  def Sample(self):
    dimensions = self.dimensions
    batchsize = self.batchsize
    numlabels = self.numlabels
    state = self.state
    sample = self.sample

    # Reshape for softmax sampling
    state.reshape((numlabels, dimensions * batchsize))
    sample.reshape((numlabels, dimensions * batchsize))

    self.state.perturb_prob_for_softmax_sampling(target=self.sample)
    self.sample.choose_max(axis=0)

    # Undo reshapes
    state.reshape((numlabels * dimensions, batchsize))
    sample.reshape((numlabels * dimensions, batchsize))

  def ComputeDeriv(self):
    """Compute derivative w.r.t input given derivative w.r.t output."""
    raise Exception('Back prop through softmax not implemented.')

  def AllocateMemory(self, batchsize):
    super(SoftmaxLayer, self).AllocateMemory(batchsize)
    self.expansion_matrix = cm.CUDAMatrix(np.eye(self.numlabels))

  def AllocateBatchsizeDependentMemory(self, batchsize):
    super(SoftmaxLayer, self).AllocateBatchsizeDependentMemory(batchsize)
    dimensions = self.dimensions
    numlabels = self.numlabels
    self.data = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))
    self.deriv = cm.CUDAMatrix(np.zeros((numlabels*dimensions, batchsize)))
    self.batchsize_temp = cm.CUDAMatrix(np.zeros((dimensions, batchsize)))

  def GetData(self):
    dimensions = self.dimensions
    batchsize = self.batchsize
    numlabels = self.numlabels
    data = self.data
    state = self.state

    # Reshape for select_columns to work
    data.reshape((1, dimensions * batchsize))  
    state.reshape((numlabels, dimensions * batchsize))
    
    # Select columns from the expansion matrix
    self.expansion_matrix.select_columns(self.data, target=self.state)

    # Undo reshapes
    data.reshape((dimensions, batchsize))
    state.reshape((numlabels * dimensions, batchsize))

  def GetLoss(self, get_deriv=False, **kwargs):
    """Compute loss and also deriv w.r.t to it if asked for.

    Compute the loss function. Targets should be in self.data, predictions
    should be in self.state.
    Args:
      get_deriv: If True, compute the derivative w.r.t the loss function and put
        it in self.deriv.
    """
    perf = deepnet_pb2.Metrics()
    perf.MergeFrom(self.proto.performance_stats)
    perf.count = self.batchsize
    tiny = self.tiny
    batchsize = self.batchsize
    dimensions = self.dimensions
    numlabels = self.numlabels
    state = self.state
    data = self.data
    temp = self.batchsize_temp
    deriv = self.deriv

    # Reshape to make each softmax be one column.
    state.reshape((numlabels, dimensions * batchsize))
    deriv.reshape((numlabels, dimensions * batchsize))
    data.reshape((1, dimensions * batchsize))
    temp.reshape((1, dimensions * batchsize))

    if self.loss_function == deepnet_pb2.Layer.CROSS_ENTROPY:
      temp = self.batchsize_temp
      
      # Compute correct predictions.
      state.get_softmax_correct(data, target=temp)
      perf.correct_preds = (temp.sum() + 0.)/ dimensions

      # Compute cross entropy.
      state.get_softmax_cross_entropy(data, target=temp, tiny=tiny)
      perf.cross_entropy = temp.sum()

      # Compute derivative.
      if get_deriv:
        state.apply_softmax_grad(data, target=self.deriv)

    elif self.loss_function == deepnet_pb2.Layer.SQUARED_LOSS:
      state.apply_softmax_grad(data, target=self.deriv)
      error = self.deriv.euclid_norm()**2
      perf.error = error
    else:
      raise Exception('Unknown loss function for Softmax units.')
    
    # Restore shapes.
    state.reshape((numlabels * dimensions, batchsize))
    deriv.reshape((numlabels * dimensions, batchsize))
    data.reshape((dimensions, batchsize))
    temp.reshape((dimensions, batchsize))
    
    return perf

  def GetSparsityDivisor(self):
    raise Exception('Sparsity not implemented for softmax units.')

