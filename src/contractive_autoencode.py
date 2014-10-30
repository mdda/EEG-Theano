"""
 ## Original code sourced from : 
 ##   https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/cA.py
 
 ## References : 
 ##   http://www-etud.iro.umontreal.ca/~rifaisal/
 ##   http://www.icml-2011.org/papers/455_icmlpaper.pdf
 
 This tutorial introduces Contractive auto-encoders (cA) using Theano.

 They are based on auto-encoders as the ones used in Bengio et
 al. 2007.  An autoencoder takes an input x and first maps it to a
 hidden representation y = f_{\theta}(x) = s(Wx+b), parameterized by
 \theta={W,b}. The resulting latent representation y is then mapped
 back to a "reconstructed" vector z \in [0,1]^d in input space z =
 g_{\theta'}(y) = s(W'y + b').  The weight matrix W' can optionally be
 constrained such that W' = W^T, in which case the autoencoder is said
 to have tied weights. The network is trained such that to minimize
 the reconstruction error (the error between x and z).  Adding the
 squared Frobenius norm of the Jacobian of the hidden mapping h with
 respect to the visible units yields the contractive auto-encoder:

      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]
      + \| \frac{\partial h(x)}{\partial x} \|^2

 References :
   - S. Rifai, P. Vincent, X. Muller, X. Glorot, Y. Bengio: Contractive
   Auto-Encoders: Explicit Invariance During Feature Extraction, ICML-11

   - S. Rifai, X. Muller, X. Glorot, G. Mesnil, Y. Bengio, and Pascal
     Vincent. Learning invariant features through local space
     contraction. Technical Report 1360, Universite de Montreal

   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import os
import sys
import time

import hickle

import numpy as np

import theano
import theano.tensor as T

#from logistic_sgd import load_data
#from utils import tile_raster_images
#
#try:
#    import PIL.Image as Image
#except ImportError:
#    import Image

class cA(object):
  """ Contractive Auto-Encoder class (cA)

  The contractive autoencoder tries to reconstruct the input with an
  additional constraint on the latent space. With the objective of
  obtaining a robust representation of the input space, we
  regularize the L2 norm(Froebenius) of the jacobian of the hidden
  representation with respect to the input. Please refer to Rifai et
  al.,2011 for more details.

  If x is the input then equation (1) computes the projection of the
  input into the latent space h. Equation (2) computes the jacobian
  of h with respect to x.  Equation (3) computes the reconstruction
  of the input, while equation (4) computes the reconstruction
  error and the added regularization term from Eq.(2).

  .. math::

      h_i = s(W_i x + b_i)                                             (1)

      J_i = h_i (1 - h_i) * W_i                                        (2)

      x' = s(W' h  + b')                                               (3)

      L = -sum_{k=1}^d [x_k \log x'_k + (1-x_k) \log( 1-x'_k)]
           + lambda * sum_{i=1}^d sum_{j=1}^n J_{ij}^2                 (4)

  """

  def __init__(self, input=None, n_visible=784, n_hidden=100,
               n_batchsize=1, W=None, bhid=None, bvis=None, numpy_rng=None):
    """Initialize the cA class by specifying the number of visible units
    (the dimension d of the input), the number of hidden units (the
    dimension d' of the latent or hidden space) and the contraction level.
    The constructor also receives symbolic variables for the input, weights
    and bias.

    :type numpy_rng: numpy.random.RandomState
    :param numpy_rng: number random generator used to generate weights

    :type input: theano.tensor.TensorType
    :param input: a symbolic description of the input or None for
                  standalone cA

    :type n_visible: int
    :param n_visible: number of visible units

    :type n_hidden: int
    :param n_hidden:  number of hidden units

    :type n_batchsize int
    :param n_batchsize: number of examples per batch

    :type W: theano.tensor.TensorType
    :param W: Theano variable pointing to a set of weights that should be
              shared belong the dA and another architecture; if dA should
              be standalone set this to None

    :type bhid: theano.tensor.TensorType
    :param bhid: Theano variable pointing to a set of biases values (for
                 hidden units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None

    :type bvis: theano.tensor.TensorType
    :param bvis: Theano variable pointing to a set of biases values (for
                 visible units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None
    """
    
    self.n_visible = n_visible
    self.n_hidden = n_hidden
    self.n_batchsize = n_batchsize
    
    # note : W' was written as `W_prime` and b' as `b_prime`
    if not W:
      print "Creating randomized cA.W"
      # W is initialized with `initial_W` which is uniformely sampled
      # from -4*sqrt(6./(n_visible+n_hidden)) and
      # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
      # converted using asarray to dtype
      # theano.config.floatX so that the code is runable on GPU
      initial_W = np.asarray(
        numpy_rng.uniform(
          low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
          high=4 * np.sqrt(6. / (n_hidden + n_visible)),
          size=(n_visible, n_hidden)
        ),
        dtype=theano.config.floatX
      )
      W = theano.shared(value=initial_W, name='W', borrow=True)

    if not bvis:
      print "Creating randomized cA.b_prime"
      bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX),
                           borrow=True)

    if not bhid:
      print "Creating randomized cA.b"
      bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX),
                           name='b',
                           borrow=True)

    self.W = W
    
    # b corresponds to the bias of the hidden
    self.b = bhid
    
    # b_prime corresponds to the bias of the visible
    self.b_prime = bvis
    
    # tied weights, therefore W_prime is W transpose
    self.W_prime = self.W.T

    # if no input is given, generate a variable representing the input
    if input is None:
      # we use a matrix because we expect a minibatch of several
      # examples, each example being a row
      self.x = T.dmatrix(name='input')
    else:
      self.x = input

    self.params = [self.W, self.b, self.b_prime]

  def get_hidden_values(self, input):
      """ Computes the values of the hidden layer """
      return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

  def get_jacobian(self, hidden, W):
      """Computes the jacobian of the hidden layer with respect to
      the input, reshapes are necessary for broadcasting the
      element-wise product on the right axis

      """
      return T.reshape(hidden * (1 - hidden),
                       (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                           W, (1, self.n_visible, self.n_hidden))

  def get_reconstructed_input(self, hidden):
      """Computes the reconstructed input given the values of the
      hidden layer

      """
      return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

  def get_cost_updates(self, contraction_level, learning_rate):
      """ This function computes the cost and the updates for one trainng
      step of the cA """

      y = self.get_hidden_values(self.x)
      z = self.get_reconstructed_input(y)
      J = self.get_jacobian(y, self.W)
      
      # note : we sum over the size of a datapoint; if we are using
      #        minibatches, L will be a vector, with one entry per
      #        example in minibatch
      self.L_rec = - T.sum(self.x * T.log(z) +
                           (1 - self.x) * T.log(1 - z),
                           axis=1)

      # Compute the jacobian and average over the number of samples/minibatch
      self.L_jacob = T.sum(J ** 2) / self.n_batchsize

      # note : L is now a vector, where each element is the
      #        cross-entropy cost of the reconstruction of the
      #        corresponding example of the minibatch. We need to
      #        compute the average of all these to get the cost of
      #        the minibatch
      cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

      # compute the gradients of the cost of the `cA` with respect
      # to its parameters
      gparams = T.grad(cost, self.params)
      
      # generate the list of updates
      updates = []
      for param, gparam in zip(self.params, gparams):
          updates.append((param, param - learning_rate * gparam))

      return (cost, updates)
  
  @classmethod
  def load_weights(_cls, f_weights):
    if not os.path.isfile(f_weights):
      return None, None, None
      
    from_hickle = hickle.load(f_weights)

    W = theano.shared(value=from_hickle['W'], name='W', borrow=True)
    b = theano.shared(value=from_hickle['b'], name='b', borrow=True)
    b_prime = theano.shared(value=from_hickle['b_prime'], name='b_prime', borrow=True)
    
    return W, b, b_prime
    
  def save_weights(self, f_weights):
    ## previously saved as :: ca.W.get_value(borrow=True)
    to_hickle = dict(
      W=self.W.get_value(borrow=True),
      b=self.b.get_value(borrow=True),
      b_prime = self.b_prime.get_value(borrow=True),
    )

    hickle.dump(to_hickle, f_weights, mode='w', compression='gzip')
    

def data_shared(data_x, borrow=True):
  """ Function that loads the dataset into shared variables

  The reason we store our dataset in shared variables is to allow
  Theano to copy it into the GPU memory (when code is run on GPU).
  Since copying data into the GPU is slow, copying a minibatch everytime
  is needed (the default behaviour if the data is not in a shared
  variable) would lead to a large decrease in performance.
  """
  shared_x = theano.shared(np.asarray(data_x.view(dtype=np.float32),
                                      dtype=theano.config.floatX),
                           borrow=borrow)
  return shared_x

def train_using_Ca(learning_rate=0.01, training_epochs=2,
                    data_x='SHARED_DATASET', 
                    input_size=None, f_weights='WEIGHTS_FILENAME', output_size=None,
                    batch_size=10, 
                    contraction_level=.1):

  """
  :type learning_rate: float
  :param learning_rate: learning rate used for training the contracting AutoEncoder

  :type training_epochs: int
  :param training_epochs: number of epochs used for training

  :type dataset: string
  :param dataset: path to the picked dataset
  """
  
  # compute number of minibatches for training, validation and testing
  n_train_batches = data_x.get_value(borrow=True).shape[0] / batch_size

  # allocate symbolic variables for the data
  index = T.lscalar()    # index to a [mini]batch
  x = T.matrix('x')      # the data is presented as a list of examples

  ####################################
  #        BUILDING THE MODEL        #
  ####################################

  W, b, b_prime = cA.load_weights(f_weights)
  
  rng = np.random.RandomState(123)

  ca = cA(
        input=x,
        n_visible=input_size, n_hidden=output_size, 
        W=W, bhid=b, bvis=b_prime,
        n_batchsize=batch_size,
        numpy_rng=rng, 
       )

  cost, updates = ca.get_cost_updates(contraction_level=contraction_level,
                                      learning_rate=learning_rate)

  train_ca = theano.function(
    [ index ],
    [ T.mean(ca.L_rec), ca.L_jacob ],
    updates=updates,
    givens={
      x: data_x[index * batch_size: (index + 1) * batch_size]
    }
  )

  print "Model Built"

  start_time = time.clock()

  # go through training epochs
  for epoch in xrange(training_epochs):
    # go through training set
    c = []
    for batch_index in xrange(n_train_batches):
      if (batch_index % 10) == 0 :
        print "Epoch %d, batch_index=%d" % (epoch, batch_index)
      c.append(train_ca(batch_index))

    c_array = np.vstack(c)
    print 'Training epoch %d, reconstruction cost ' % epoch, np.mean(
      c_array[0]), ' jacobian norm ', np.mean(np.sqrt(c_array[1]))

  end_time = time.clock()

  training_time = (end_time - start_time)

  print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((training_time) / 60.))
                        
  #image = Image.fromarray(tile_raster_images(
  #    X=ca.W.get_value(borrow=True).T,
  #    img_shape=(28, 28), tile_shape=(10, 10),
  #    tile_spacing=(1, 1)))
  #
  
  ## Save weight matrix  
  ca.save_weights(f_weights)


def test_using_Ca(data_x='FILL_IN_DATASET', f_weights='WEIGHTS_FILENAME', f_output='OUTPUT_FILENAME'):
  n_train_batches = data_x.get_value(borrow=True).shape[0] / batch_size

  # allocate symbolic variables for the data
  index = T.lscalar()    # index to a [mini]batch
  x = T.matrix('x')      # the data is presented as a list of examples

  W, b, b_prime = cA.load_weights(f_weights)
  
  ca = cA(
        input=x,
        #n_visible=input_size, n_hidden=output_size, # Not necessary - weights are loaded
        W=W, bhid=b, bvis=b_prime,
        n_batchsize=batch_size,
       )

  test_ca = theano.function(
    [ index ],
    #[ T.mean(ca.L_rec), ca.L_jacob ],
    ca.get_hidden_values(ca.x),
    #updates=updates,
    givens={
      x: data_x[index * batch_size: (index + 1) * batch_size]
    }
  )

  c = []
  for batch_index in xrange(n_train_batches):
    c.append(test_ca(batch_index))

  c_array = np.vstack(c)
  #print 'Testing epoch %d, reconstruction cost ' % epoch, np.mean(
  #  c_array[0]), ' jacobian norm ', np.mean(np.sqrt(c_array[1]))
  
  np.shape(c_array)

  
if __name__ == '__main__':
  
  _patient = 'Dog_2'
  _patient = 'Patient_2'
  
  train_data = True # and False
  
  input_size  = None # i.e. determine from f_in
  output_size = 200  # Need some number to start us off
  
  ## Two modes : Test and Train
  f_in  = "data/feat/%s/%s_%s_input.hickle" % (_patient, _patient, ("train" if train_data else "test"), )
  
  f_weights = "data/layer1_feat-200/%s/%s_weights.hickle" % (_patient, _patient,)
  
  f_out = "data/layer1_feat-200/%s/%s_%s_hidden.hickle" % (_patient, _patient, ("train" if train_data else "test"), )
  
  ## Load input file
  layer_previous = hickle.load(f_in)
  data_x = data_shared(layer_previous['features'])
  # TODO: something with timestamps array too...
  
  #print "input features shape (complex) : ", np.shape(layer_previous['features'])
  #input_size = np.shape(layer_previous['features'])[1]
  
  print "input features shape (theano)  : ", np.shape(data_x.get_value(borrow=True))
  input_size = np.shape(data_x.get_value(borrow=True))[1]
  
  if train_data:
    train_using_Ca(data_x = data_x, input_size=input_size, f_weights=f_weights, output_size=output_size)
  else:
    test_using_Ca(data_x=data_x, f_weights=f_weights, f_output=f_out)
