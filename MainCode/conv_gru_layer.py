# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:29:49 2024

@author: user
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, InputSpec

class ConvGRU2D(Layer):
    def __init__(self, filters, kernel_size, padding='same', activation='tanh', recurrent_activation='sigmoid', **kwargs):
        super(ConvGRU2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.state_size = [filters, filters]

    def build(self, input_shape):
        self.batch_size, self.time_steps, self.height, self.width, self.channels = input_shape
        self.Wz = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=self.recurrent_activation)
        self.Wr = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=self.recurrent_activation)
        self.Wh = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=None)

        self.Uz = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=self.recurrent_activation)
        self.Ur = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=self.recurrent_activation)
        self.Uh = Conv2D(self.filters, self.kernel_size, padding=self.padding, activation=None)

        super(ConvGRU2D, self).build(input_shape)

    def call(self, inputs, states=None):
        h_prev = states[0] if states is not None else tf.zeros_like(inputs[:, 0, :, :, :])
        output_sequence = []

        for t in range(self.time_steps):
            x_t = inputs[:, t, :, :, :]
            z_t = self.recurrent_activation(self.Wz(x_t) + self.Uz(h_prev))
            r_t = self.recurrent_activation(self.Wr(x_t) + self.Ur(h_prev))
            h_hat_t = self.activation(self.Wh(x_t) + self.Uh(r_t * h_prev))
            h_t = (1 - z_t) * h_prev + z_t * h_hat_t
            output_sequence.append(h_t)
            h_prev = h_t

        output_sequence = tf.stack(output_sequence, axis=1)
        return output_sequence, h_t

    def get_config(self):
        config = super(ConvGRU2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation),
            'recurrent_activation': tf.keras.activations.serialize(self.recurrent_activation)
        })
        return config