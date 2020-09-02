#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from NumPyNet.layers import RNN_layer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestRNNLayer:
    '''
    Tests:
      - costructor of RNN_layer object
      - print function

    to be:
       forward function against tf.keras
       update function
       backward function against tf.keras
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
                RNN_layer(outputs=outputs, steps=steps)

            outputs += 10
            weights_choice = [[np.random.uniform(low=-1, high=1., size=(w * h * c, outputs))] * 3, None]
            bias_choice = [[np.random.uniform(low=-1, high=1., size=(outputs,))] * 3, None]

        weights = choice(weights_choice)
        bias = choice(bias_choice)

        for numpynet_act in numpynet_activ:
            layer = RNN_layer(outputs=outputs, steps=steps, activation=numpynet_act,
                              input_shape=(b, w, h, c),
                              weights=weights, bias=bias)

            if weights is not None:
                np.testing.assert_allclose(layer.input_layer.weights, weights[0], rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(layer.self_layer.weights, weights[1], rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(layer.output_layer.weights, weights[2], rtol=1e-5, atol=1e-8)

            if bias is not None:
                np.testing.assert_allclose(layer.input_layer.bias, bias[0], rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(layer.self_layer.bias, bias[1], rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(layer.output_layer.bias, bias[2], rtol=1e-5, atol=1e-8)

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

        layer = RNN_layer(outputs=outputs, steps=steps, activation='linear')

        with pytest.raises(TypeError):
            print(layer)

        layer = RNN_layer(outputs=outputs, steps=steps, activation='linear', input_shape=(b, w, h, c))

        print(layer)
