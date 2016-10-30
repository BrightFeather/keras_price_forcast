from keras.layers import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import time_distributed_dense
from __future__ import absolute_import
import numpy as np

class ESN(Recurrent):
    '''Echo State Network - Herbert Jaeger 2007

    Description of the algorithm, see:
    [this document](www.scholarpedia.org/article/Echo_State_Network)
    Library, articles and code on this algorithms:
    [this webpage](reservoir-computing.org)

    Using parameters on this paper
    [Financial Market Time Series Prediction with Recurrent Neural Networks]
    (cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTime SeriesPredictionwithRecurrentNeural.pdf)

    # Arguments
        utput_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [The "echo state" approach to analysing and training recurrent neural networks-with an eratum note, Technical report GMD report, 148, 2001]
        - [Tutorial on training recurrent neural networks covering BPPT, RTRL, RKL and the "echo state network approach"]
    '''

def __init__(self, channel_dim, output_dim,
             init='glorot_uniform', inner_init='orthogonal',
             forget_bias_init='one', activation='hard_sigmoid',
             inner_activation='tanh', W_regularizer=None,
             U_regularizer=None, b_regularizer=None,
             dropout_W=0., dropout_U=0., **kwargs ):
    self.channel_dim = channel_dim
    self.output_dim = output_dim
    self.init = initializations.get(init)
    self.inner_init = initializations.get(inner_init)
    self.forget_bias_init = initializations.get(forget_bias_init)
    self.activation = activations.get(activation)
    self.inner_activation = activations.get(inner_activation)
    self.W_regularizer = regularizers.get(W_regularizer)
    self.U_regularizer = regularizers.get(U_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.dropout_W, self.dropout_U = dropout_W, dropout_U

    if self.dropout_W or self.dropout_U:
        self.uses_learning_phase = True
    super(ESN, self).__init__(**kwargs)

def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape)]
    self.input_dim = input_shape[2]


    if self.stateful:
        self.reset_states()
    else:
        self.states = [None, None]
    if self.consume_less == 'gpu':
        self.W_in = self.init((self.input_dim, 4 * self.output_dim),
                           name='{}_W'.format(self.name))
        self.W_in = self.inner_init((self.output_dim, 4 * self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                       K.get_value(self.forget_bias_init(self.output_dim)),
                                       np.zeros(self.output_dim),
                                       np.zeros(self.output_dim))),
                            name='{}_b'.format(self.name))
        self.trainable_weights = [self.W]
    else:
        self.W_in = self.init((self.input_dim, self.channel_dim),
                             name='{}_W_in'.format(self.name))
        self.W = self.inner_init((self.channel_dim, self.channel_dim),
                                   name='{}_W'.format(self.name))
        self.W_fb = K.inner_init((self.output_dim, self.channel_dim,),
                             name='{}_W_out'.format(self.name))
        self.W_out = K.inner_init(((self.input_dim + self.channel_dim,),self.output_dim),
                             name='{}_W_out'.format(self.name))
        self.b_f =  self.forget_bias_init((self.channel_dim,),
                                         name='{}_b_f'.format(self.name))
        self.b_out =  self.forget_bias_init((self.output_dim,),
                                         name='{}_b_out'.format(self.name))
        self.trainable_weights = [self.W_out, self.b_out ]

    self.regularizers = []
    if self.W_out_regularizer:
        self.W_out_regularizer.set_param(self.W_out)
        self.regularizers.append(self.W_out_regularizer)
    if self.b_out_regularizer:
        self.b_out_regularizer.set_param(self.b_out)
        self.regularizers.append(self.b_out_regularizer)

    if self.initial_weights is not None:
        self.set_weights(self.initial_weights)
        del self.initial_weights

    def reset_states(self): # needs further edition
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x = time_distributed_dense(x, self.W_out, self.b_out, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            # x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
            #                              input_dim, self.output_dim, timesteps)
            # x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
            #                              input_dim, self.output_dim, timesteps)
            return x
        else:
            return x

    def step(self, input, states):
        prev_output = states[0]
        # B_U = states[1]
        B_W_in = states[1]

        z = K.concatenate(prev_output,input,axis=-1)
        y = self.activation(K.dot(self.W_out, z  + self.b_out))
        x = self.inner_activation(K.dot(self.W_in, input) + K.dot(self.W, prev_output) + K.dot(self.W_fb,y ) + self.b)

        return y, [y, x]






    def get_constants(self, x): # needs further edition
        constants = []
        # if 0 < self.dropout_U < 1:
        #     ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
        #     ones = K.concatenate([ones] * self.output_dim, 1)
        #     B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
        #     constants.append(B_U)
        # else:
        #     constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self): # needs further edition
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  # 'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  # 'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}

        base_config = super(ESN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
