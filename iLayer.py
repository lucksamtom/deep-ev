#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.engine.topology import Layer
import numpy
from keras import backend as K

class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = numpy.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def compute_output_shape(self, input_shape):
        return input_shape