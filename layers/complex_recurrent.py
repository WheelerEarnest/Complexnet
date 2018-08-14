# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
#
# ------------------------------------------------------------
import numpy as np
import keras.backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec, RNN, SimpleRNN
from util import c_elem_mult

# TODO: create more robust initializers for complex valued weights

class SimpleCRNN(SimpleRNN):
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        # Note: theano backend does not support dropout
        cell = SimpleCRNNCell(units,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout)
        super(SimpleCRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

class SimpleCRNNCell(Layer):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SimpleCRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        si
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units * 2
        # Todo: implement a dropout mask that properly drops complex values
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):

        self.real_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                           name='real_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.imaginary_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                                name='imaginary_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)

        self.recurrent_real_kernel = self.add_weight(shape=(self.units, self.units),
                                                     name='recurrent_real_kernel',
                                                     initializer=self.recurrent_initializer,
                                                     regularizer=self.recurrent_regularizer,
                                                     constraint=self.recurrent_constraint)
        self.recurrent_imaginary_kernel = self.add_weight(shape=(self.units, self.units),
                                                          name='recurrent_imaginary_kernel',
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0]
        # Todo: properly implement dropout masks to accomidate the complex values

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        cat_kernel = K.concatenate(
            [K.concatenate([self.real_kernel, -self.imaginary_kernel], axis=-1),
             K.concatenate([self.imaginary_kernel, self.real_kernel], axis=-1)],
            axis=0
        )

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, cat_kernel)
        else:
            h = K.dot(inputs, cat_kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask

        cat_rec_kernel = K.concatenate(
            [K.concatenate([self.recurrent_real_kernel, -self.recurrent_imaginary_kernel], axis=-1),
             K.concatenate([self.recurrent_imaginary_kernel, self.recurrent_real_kernel], axis=-1)],
            axis=0
        )
        output = h + K.dot(prev_output, cat_rec_kernel)
        if self.activation is not None:
            output = self.activation(output)

        # Set learning phase on output tensor
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleCRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CGRUCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(CGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        # Remember: the state size is doubled because of the imaginary component
        self.state_size = self.units * 2
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # We need to make sure that the input is in the correct complex format
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1]
        self.real_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                           name='real_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.imaginary_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                                name='imaginary_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer= self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.real_recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3),
                                                     name='recurrent_real_kernel',
                                                     initializer=self.recurrent_initializer,
                                                     regularizer=self.recurrent_regularizer,
                                                     constraint=self.recurrent_constraint)
        self.imaginary_recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3),
                                                          name='recurrent_imaginary_kernel',
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint)
        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.real_bias = self.add_weight(shape=bias_shape,
                                             name='real_bias',
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.imaginary_bias = self.add_weight(shape=bias_shape,
                                                  name='imaginary_bias',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
            if not self.reset_after:
                self.real_input_bias, self.real_recurrent_bias = self.real_bias, None
                self.imaginary_input_bias, self.imaginary_recurrent_bias = self.imaginary_bias, None
            else:
                self.real_input_bias = K.flatten(self.real_bias[0])
                self.imaginary_input_bias = K.flatten(self.imaginary_bias[0])
                self.real_recurrent_bias = K.flatten(self.real_bias[1])
                self.imaginary_recurrent_bias = K.flatten(self.imaginary_bias[1])
        else:
            self.bias = None

        # update gate
        self.real_kernel_z = self.real_kernel[:, :self.units]
        self.imaginary_kernel_z = self.imaginary_kernel[:, :self.units]
        self.real_recurrent_kernel_z = self.real_recurrent_kernel[:, :self.units]
        self.imaginary_recurrent_kernel_z = self.imaginary_recurrent_kernel[:, :self.units]
        # reset gate
        self.real_kernel_r = self.real_kernel[:, self.units: self.units * 2]
        self.imaginary_kernel_r = self.imaginary_kernel[:, self.units: self.units * 2]
        self.real_recurrent_kernel_r = self.real_recurrent_kernel[:,
                                       self.units:
                                       self.units * 2]
        self.imaginary_recurrent_kernel_r = self.imaginary_recurrent_kernel[:,
                                            self.units:
                                            self.units * 2]
        # new gate
        self.real_kernel_h = self.real_kernel[:, self.units * 2:]
        self.imaginary_kernel_h = self.imaginary_kernel[:, self.units * 2:]
        self.real_recurrent_kernel_h = self.real_recurrent_kernel[:, self.units * 2:]
        self.imaginary_recurrent_kernel_h = self.imaginary_recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.real_input_bias_z = self.real_input_bias[:self.units]
            self.imaginary_input_bias_z = self.imaginary_input_bias[:self.units]
            self.real_input_bias_r = self.real_input_bias[self.units: self.units * 2]
            self.imaginary_input_bias_r = self.imaginary_input_bias[self.units: self.units * 2]
            self.real_input_bias_h = self.real_input_bias[self.units * 2:]
            self.imaginary_input_bias_h = self.imaginary_input_bias[self.units * 2:]
            if self.reset_after:
                self.real_recurrent_bias_z = self.real_recurrent_bias[:self.units]
                self.imaginary_recurrent_bias_z = self.imaginary_recurrent_bias[:self.units]
                self.real_recurrent_bias_r = self.real_recurrent_bias[self.units: self.units * 2]
                self.imaginary_recurrent_bias_r = self.imaginary_recurrent_bias[self.units: self.units * 2]
                self.real_recurrent_bias_h = self.real_recurrent_bias[self.units * 2:]
                self.imaginary_recurrent_bias_h = self.imaginary_recurrent_bias[self.units * 2:]

        else:
            self.real_input_bias_z = None
            self.imaginary_input_bias_z = None
            self.real_input_bias_r = None
            self.imaginary_input_bias_r = None
            self.real_input_bias_h = None
            self.imaginary_input_bias_h = None
            if self.reset_after:
                self.real_recurrent_bias_z = None
                self.imaginary_recurrent_bias_z = None
                self.real_recurrent_bias_r = None
                self.imaginary_recurrent_bias_r = None
                self.real_recurrent_bias_h = None
                self.imaginary_recurrent_bias_h = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]

        #Todo: properly implement dropout masks

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        # Currently only implementation 1 is written
        if self.implementation is not None:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            cat_kernel_z = K.concatenate(
                [K.concatenate([self.real_kernel_z, -self.imaginary_kernel_z], axis=-1),
                 K.concatenate([self.imaginary_kernel_z, self.real_kernel_z], axis=-1)],
                axis=0
            )
            cat_kernel_r = K.concatenate(
                [K.concatenate([self.real_kernel_r, -self.imaginary_kernel_r], axis=-1),
                 K.concatenate([self.imaginary_kernel_r, self.real_kernel_r], axis=-1)],
                axis=0
            )
            cat_kernel_h = K.concatenate(
                [K.concatenate([self.real_kernel_h, -self.imaginary_kernel_h], axis=-1),
                 K.concatenate([self.imaginary_kernel_h, self.real_kernel_h], axis=-1)],
                axis=0
            )
            x_z = K.dot(inputs_z, cat_kernel_z)
            x_r = K.dot(inputs_r, cat_kernel_r)
            x_h = K.dot(inputs_h, cat_kernel_h)
            cat_bias_z = None
            cat_bias_r = None
            cat_bias_h = None
            if self.use_bias:
                cat_bias_z = K.concatenate([self.real_input_bias_z, self.imaginary_input_bias_z],
                                      axis=0)
                x_z = K.bias_add(x_z, cat_bias_z)
                cat_bias_r = K.concatenate([self.real_input_bias_r, self.imaginary_input_bias_r],
                                           axis=0)
                x_r = K.bias_add(x_r, cat_bias_r)
                cat_bias_h = K.concatenate([self.real_input_bias_h, self.imaginary_input_bias_h],
                                           axis=0)
                x_h = K.bias_add(x_h, cat_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            cat_recurrent_kernel_z = K.concatenate(
                [K.concatenate([self.real_recurrent_kernel_z, -self.imaginary_recurrent_kernel_z], axis=-1),
                 K.concatenate([self.imaginary_recurrent_kernel_z, self.real_recurrent_kernel_z], axis=-1)],
                axis=0
            )
            recurrent_z = K.dot(h_tm1_z, cat_recurrent_kernel_z)
            cat_recurrent_kernel_r = K.concatenate(
                [K.concatenate([self.real_recurrent_kernel_r, -self.imaginary_recurrent_kernel_r], axis=-1),
                 K.concatenate([self.imaginary_recurrent_kernel_r, self.real_recurrent_kernel_r], axis=-1)],
                axis=0
            )
            recurrent_r = K.dot(h_tm1_r, cat_recurrent_kernel_r)
            if self.reset_after and self.use_bias:
                cat_recurrent_bias_z = K.concat([self.real_recurrent_bias_z, self.imaginary_recurrent_bias_z],
                                                axis=0)
                recurrent_z = K.bias_add(recurrent_z, cat_recurrent_bias_z)
                cat_recurrent_bias_r = K.concat([self.real_recurrent_bias_r, self.imaginary_recurrent_bias_r],
                                                axis=0)
                recurrent_r = K.bias_add(recurrent_r, cat_recurrent_bias_r)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            cat_recurrent_kernel_h = K.concatenate(
                [K.concatenate([self.real_recurrent_kernel_h, -self.imaginary_recurrent_kernel_h],
                               axis=-1),
                 K.concatenate([self.imaginary_recurrent_kernel_h, self.real_recurrent_kernel_h])],
                axis=0
            )
            # reset gate applied before/after matrix mult
            if self.reset_after:

                recurrent_h = K.dot(h_tm1_h, cat_recurrent_kernel_h)
                if self.use_bias:
                    cat_recurrent_bias_h = K.concat([self.real_recurrent_bias_h, self.imaginary_recurrent_bias_h],
                                                    axis=0)
                    recurrent_h = K.bias_add(recurrent_h, cat_recurrent_bias_h)
                recurrent_h = c_elem_mult(r, recurrent_h, self.units)
            else:
                recurrent_h = K.dot(c_elem_mult(r, h_tm1_h, self.units), cat_recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h)
        else:
            # Todo: implement implementation 2 (as per the Keras code)
            None
        h = c_elem_mult(z, h_tm1, self.units) + c_elem_mult(1 - z, hh, self.units)

        if 0. < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(GRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))