#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from tensorflow.keras.layers import SimpleRNN, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from NumPyNet.layers.simple_rnn_layer import SimpleRNN_layer
from NumPyNet.optimizer import Momentum
from NumPyNet.utils import data_to_timesteps

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestSimpleRNNLayer:
    '''
    Tests:
      - costructor of SimpleRNN_layer object
      - print function
      - forward function against tf.keras
      - backward + update function against tf.keras

      TODO backward: return_sequence=True, delta_keras
    '''

    @given(outputs=st.integers(min_value=-3, max_value=10),
           steps=st.integers(min_value=1, max_value=4),
           b=st.integers(min_value=5, max_value=15),
           w=st.integers(min_value=15, max_value=100),
           h=st.integers(min_value=15, max_value=100),
           c=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10,
              deadline=None)
    def test_constructor(self, outputs, steps, b, w, h, c):

        numpynet_activ = ['relu', 'logistic', 'tanh', 'linear']

        if outputs > 0:
            weights_choice = [np.random.uniform(low=-1, high=1., size=(w * h * c, outputs)), None]
            bias_choice = [np.random.uniform(low=-1, high=1., size=(outputs,)), None]

        else:
            with pytest.raises(ValueError):
                SimpleRNN_layer(outputs=outputs, steps=steps)

            outputs += 10
            weights_choice = [[np.random.uniform(low=-1, high=1., size=(w * h * c, outputs))] * 3, None]
            bias_choice = [[np.random.uniform(low=-1, high=1., size=(outputs,))] * 3, None]

        weights = choice(weights_choice)
        bias = choice(bias_choice)

        for numpynet_act in numpynet_activ:
            layer = SimpleRNN_layer(outputs=outputs, steps=steps, activation=numpynet_act,
                                    input_shape=(b, w, h, c),
                                    weights=weights, bias=bias)

            assert layer.output is None

    @given(outputs=st.integers(min_value=3, max_value=10),
           steps=st.integers(min_value=1, max_value=4),
           b=st.integers(min_value=5, max_value=15),
           w=st.integers(min_value=15, max_value=100),
           h=st.integers(min_value=15, max_value=100),
           c=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10,
              deadline=None)
    def test_printer(self, outputs, steps, b, w, h, c):

        layer = SimpleRNN_layer(outputs=outputs, steps=steps, activation='linear')

        with pytest.raises(AttributeError):
            print(layer)

        layer = SimpleRNN_layer(outputs=outputs, steps=steps, activation='linear', input_shape=(b, w, h, c))

        print(layer)

    @staticmethod
    def initialize_step(steps, outputs, features, batch, return_seq, activation, inpt_keras):
        # weights init
        kernel = np.random.uniform(low=-1, high=1, size=(features, outputs))
        recurrent_kernel = np.random.uniform(low=-1, high=1, size=(outputs, outputs))
        bias = np.random.uniform(low=-1, high=1, size=(outputs,))

        # create keras model
        inp = Input(shape=inpt_keras.shape[1:])
        rnn = SimpleRNN(units=outputs, activation=activation, return_sequences=return_seq)(inp)
        model = Model(inputs=inp, outputs=rnn)

        # set weights for the keras model
        model.set_weights([kernel, recurrent_kernel, bias])

        # create NumPyNet layer
        layer = SimpleRNN_layer(outputs=outputs, steps=steps, input_shape=(batch, 1, 1, features),
                                activation=activation, return_sequence=return_seq)

        # set NumPyNet weights
        layer.load_weights(np.concatenate([bias.ravel(), kernel.ravel(), recurrent_kernel.ravel()]))

        return model, layer

    @staticmethod
    def forward_step(model, layer, inpt_keras, inpt):
        # forward for keras
        forward_out_keras = model.predict(inpt_keras)

        # forward NumPyNet
        layer.forward(inpt)
        forward_out_numpynet = layer.output.reshape(forward_out_keras.shape)
        return forward_out_numpynet, forward_out_keras

    @given(steps=st.integers(min_value=1, max_value=10),
           outputs=st.integers(min_value=1, max_value=50),
           features=st.integers(min_value=1, max_value=50),
           batch=st.integers(min_value=20, max_value=100),
           return_seq=st.booleans())
    @settings(max_examples=10, deadline=None)
    def test_forward(self, steps, outputs, features, batch, return_seq):

        activation = 'tanh'

        inpt = np.random.uniform(size=(batch, features))
        inpt_keras, _ = data_to_timesteps(inpt, steps=steps)

        assert inpt_keras.shape == (batch - steps, steps, features)

        # INITIALIZE
        model, layer = TestSimpleRNNLayer.initialize_step(steps, outputs, features, batch, return_seq, activation,
                                                          inpt_keras)

        # FORWARD
        forward_out_numpynet, forward_out_keras = TestSimpleRNNLayer.forward_step(model, layer, inpt_keras, inpt)

        assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-4, rtol=1e-3)

    @given(outputs=st.integers(min_value=3, max_value=10),
           steps=st.integers(min_value=1, max_value=4),
           features=st.integers(min_value=1, max_value=50),
           batch=st.integers(min_value=20, max_value=100),
           return_seq=st.booleans())
    @settings(max_examples=10,
              deadline=None)
    def test_backprop(self, outputs, steps, features, batch, return_seq):

        activation = 'tanh'
        return_seq = False

        with tf.Graph().as_default():
            inpt = np.random.uniform(size=(batch, features))
            inpt_keras, _ = data_to_timesteps(inpt, steps=steps)

            # INITIALIZE
            model, layer = TestSimpleRNNLayer.initialize_step(steps, outputs, features, batch, return_seq, activation,
                                                              inpt_keras)

            # FORWARD
            TestSimpleRNNLayer.forward_step(model, layer, inpt_keras, inpt)

            lr = .1
            momentum = .9
            decay = 0.
            layer.optimizer = Momentum(lr=lr, momentum=momentum, decay=decay)
            model.optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=False, decay=decay)

            # BACKWARD
            print('outputs ' + str(outputs))
            print('steps ' + str(steps))
            print('features ' + str(features))
            print('batch ' + str(batch))
            print('return_seq ' + str(return_seq))
            layer.delta = np.ones(shape=layer.out_shape, dtype=float)
            a, b, c = layer.X.shape
            delta = np.ones(shape=(b, a, c), dtype=float)
            layer.backward(delta=delta, copy=True)

            tf.compat.v1.disable_eager_execution()
            # Compute the gradient of output w.r.t the weights
            gradient = K.gradients(model.output, model.trainable_weights)

            # Define a function to evaluate the gradient
            func = K.function(model.inputs + model.trainable_weights + model.outputs, gradient)

            updates = func([inpt_keras])
            weights_update_keras = updates[0]
            recurrent_weights_update_keras = updates[1]
            bias_update_keras = updates[2]

            np.testing.assert_allclose(layer.bias_update, bias_update_keras, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(layer.weights_update, weights_update_keras, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(layer.recurrent_weights_update, recurrent_weights_update_keras, atol=1e-4,
                                       rtol=1e-4)
