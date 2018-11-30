# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
#
# ------------------------------------------------------------
import warnings

import keras.backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, RNN
from keras.layers.recurrent import _generate_dropout_mask

from complexnet.util import c_elem_mult


# TODO: create more robust initializers for complex valued weights
class SimpleCRNN(RNN):
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

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(SimpleCRNN, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)
    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleCRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


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
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units * 2
        self.output_size = self.units * 2
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        assert input_shape % 2 == 0
        input_dim = input_shape[-1] // 2

        self.real_kernel = self.add_weight(shape=(input_dim, self.units),
                                           name='real_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.imaginary_kernel = self.add_weight(shape=(input_dim, self.units),
                                                name='imaginary_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.cat_kernel = K.concatenate(
            [K.concatenate([self.real_kernel, -self.imaginary_kernel], axis=-1),
             K.concatenate([self.imaginary_kernel, self.real_kernel], axis=-1)],
            axis=0
        )

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
        self.cat_rec_kernel = K.concatenate(
            [K.concatenate([self.recurrent_real_kernel, -self.recurrent_imaginary_kernel], axis=-1),
             K.concatenate([self.recurrent_imaginary_kernel, self.recurrent_real_kernel], axis=-1)],
            axis=0
        )
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
        # Dropout is a bit tricky here: we really only want to create a mask for half of the input and then duplicate and concatenate it,
        # because of how complex numbers are represented here. In other words, we make a mask for the real component and then
        # duplicate it for the imaginary component
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            input_size = K.int_shape(inputs)[-1] // 2
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs[:, :input_size]),
                self.dropout,
                training=training,
                count=4)
            self._dropout_mask = [K.concatenate([mask, mask]) for mask in self._dropout_mask]
        if (0 < self.recurrent_dropout < 1 and
            self._recurrent_dropout_mask is None):
            state_size = K.int_shape(states[0])[-1] // 2
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0][:, :state_size]),
                self.recurrent_dropout,
                training=training,
                count=4
            )
            self._recurrent_dropout_mask = [K.concatenate([mask, mask]) for mask in self._recurrent_dropout_mask]

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask


        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.cat_kernel)
        else:
            h = K.dot(inputs, self.cat_kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask


        output = h + K.dot(prev_output, self.cat_rec_kernel)
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
        self.output_size = self.units * 2
        self.state_size = self.units * 2
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # We need to make sure that the input is in the correct complex format
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        self.real_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                           name='real_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.imaginary_kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                                name='imaginary_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
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
        self.cat_kernel_z = K.concatenate(
            [K.concatenate([self.real_kernel_z, -self.imaginary_kernel_z], axis=-1),
             K.concatenate([self.imaginary_kernel_z, self.real_kernel_z], axis=-1)],
            axis=0
        )
        self.real_recurrent_kernel_z = self.real_recurrent_kernel[:, :self.units]
        self.imaginary_recurrent_kernel_z = self.imaginary_recurrent_kernel[:, :self.units]
        self.cat_recurrent_kernel_z = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_z, -self.imaginary_recurrent_kernel_z], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_z, self.real_recurrent_kernel_z], axis=-1)],
            axis=0
        )
        # reset gate
        self.real_kernel_r = self.real_kernel[:, self.units: self.units * 2]
        self.imaginary_kernel_r = self.imaginary_kernel[:, self.units: self.units * 2]
        self.cat_kernel_r = K.concatenate(
            [K.concatenate([self.real_kernel_r, -self.imaginary_kernel_r], axis=-1),
             K.concatenate([self.imaginary_kernel_r, self.real_kernel_r], axis=-1)],
            axis=0
        )
        self.real_recurrent_kernel_r = self.real_recurrent_kernel[:,
                                       self.units:
                                       self.units * 2]
        self.imaginary_recurrent_kernel_r = self.imaginary_recurrent_kernel[:,
                                            self.units:
                                            self.units * 2]
        self.cat_recurrent_kernel_r = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_r, -self.imaginary_recurrent_kernel_r], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_r, self.real_recurrent_kernel_r], axis=-1)],
            axis=0
        )
        # new gate
        self.real_kernel_h = self.real_kernel[:, self.units * 2:]
        self.imaginary_kernel_h = self.imaginary_kernel[:, self.units * 2:]
        self.cat_kernel_h = K.concatenate(
            [K.concatenate([self.real_kernel_h, -self.imaginary_kernel_h], axis=-1),
             K.concatenate([self.imaginary_kernel_h, self.real_kernel_h], axis=-1)],
            axis=0
        )
        self.real_recurrent_kernel_h = self.real_recurrent_kernel[:, self.units * 2:]
        self.imaginary_recurrent_kernel_h = self.imaginary_recurrent_kernel[:, self.units * 2:]
        self.cat_recurrent_kernel_h = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_h, -self.imaginary_recurrent_kernel_h],
                           axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_h, self.real_recurrent_kernel_h])],
            axis=0
        )
        if self.use_bias:
            # bias for inputs
            self.real_input_bias_z = self.real_input_bias[:self.units]
            self.imaginary_input_bias_z = self.imaginary_input_bias[:self.units]
            self.cat_bias_z = K.concatenate([self.real_input_bias_z, self.imaginary_input_bias_z],
                                       axis=0)
            self.real_input_bias_r = self.real_input_bias[self.units: self.units * 2]
            self.imaginary_input_bias_r = self.imaginary_input_bias[self.units: self.units * 2]
            self.cat_bias_r = K.concatenate([self.real_input_bias_r, self.imaginary_input_bias_r],
                                       axis=0)
            self.real_input_bias_h = self.real_input_bias[self.units * 2:]
            self.imaginary_input_bias_h = self.imaginary_input_bias[self.units * 2:]
            self.cat_bias_h = K.concatenate([self.real_input_bias_h, self.imaginary_input_bias_h],
                                       axis=0)
            if self.reset_after:
                self.real_recurrent_bias_z = self.real_recurrent_bias[:self.units]
                self.imaginary_recurrent_bias_z = self.imaginary_recurrent_bias[:self.units]
                self.cat_recurrent_bias_z = K.concat([self.real_recurrent_bias_z, self.imaginary_recurrent_bias_z],
                                                axis=0)
                self.real_recurrent_bias_r = self.real_recurrent_bias[self.units: self.units * 2]
                self.imaginary_recurrent_bias_r = self.imaginary_recurrent_bias[self.units: self.units * 2]
                self.cat_recurrent_bias_r = K.concat([self.real_recurrent_bias_r, self.imaginary_recurrent_bias_r],
                                                axis=0)
                self.real_recurrent_bias_h = self.real_recurrent_bias[self.units * 2:]
                self.imaginary_recurrent_bias_h = self.imaginary_recurrent_bias[self.units * 2:]
                self.cat_recurrent_bias_h = K.concat([self.real_recurrent_bias_h, self.imaginary_recurrent_bias_h],
                                                axis=0)

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

        # Dropout is a bit tricky here: we really only want to create a mask for half of the input and then duplicate and concatenate it,
        # because of how complex numbers are represented here. In other words, we make a mask for the real component and then
        # duplicate it for the imaginary component
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            input_size = K.int_shape(inputs)[-1] // 2
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs[:, :input_size]),
                self.dropout,
                training=training,
                count=4)
            self._dropout_mask = [K.concatenate([mask, mask]) for mask in self._dropout_mask]
        if (0 < self.recurrent_dropout < 1 and
            self._recurrent_dropout_mask is None):
            state_size = K.int_shape(states[0])[-1] // 2
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0][:, :state_size]),
                self.recurrent_dropout,
                training=training,
                count=4
            )
            self._recurrent_dropout_mask = [K.concatenate([mask, mask]) for mask in self._recurrent_dropout_mask]

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


            x_z = K.dot(inputs_z, self.cat_kernel_z)
            x_r = K.dot(inputs_r, self.cat_kernel_r)
            x_h = K.dot(inputs_h, self.cat_kernel_h)
            cat_bias_z = None
            cat_bias_r = None
            cat_bias_h = None
            if self.use_bias:

                x_z = K.bias_add(x_z, self.cat_bias_z)

                x_r = K.bias_add(x_r, self.cat_bias_r)

                x_h = K.bias_add(x_h, self.cat_bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1


            recurrent_z = K.dot(h_tm1_z, self.cat_recurrent_kernel_z)

            recurrent_r = K.dot(h_tm1_r, self.cat_recurrent_kernel_r)
            if self.reset_after and self.use_bias:

                recurrent_z = K.bias_add(recurrent_z, self.cat_recurrent_bias_z)

                recurrent_r = K.bias_add(recurrent_r, self.cat_recurrent_bias_r)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied before/after matrix mult
            if self.reset_after:

                recurrent_h = K.dot(h_tm1_h, self.cat_recurrent_kernel_h)
                if self.use_bias:

                    recurrent_h = K.bias_add(recurrent_h, self.cat_recurrent_bias_h)
                recurrent_h = c_elem_mult(r, recurrent_h, self.units)
            else:
                recurrent_h = K.dot(c_elem_mult(r, h_tm1_h, self.units), self.cat_recurrent_kernel_h)

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
        base_config = super(CGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CGRU(RNN):
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
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):
        if implementation == 2:
            warnings.warn('`implementation=2 has not been implemented.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = CGRUCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
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
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation,
                        reset_after=reset_after)
        super(CGRU, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        def call(self, inputs, mask=None, training=None, initial_state=None):
            self.cell._dropout_mask = None
            self.cell._recurrent_dropout_mask = None
            return super(CGRU, self).call(inputs,
                                                mask=mask,
                                                training=training,
                                                initial_state=initial_state)

        @property
        def units(self):
            return self.cell.units

        @property
        def activation(self):
            return self.cell.activation

        @property
        def use_bias(self):
            return self.cell.use_bias

        @property
        def kernel_initializer(self):
            return self.cell.kernel_initializer

        @property
        def recurrent_initializer(self):
            return self.cell.recurrent_initializer

        @property
        def bias_initializer(self):
            return self.cell.bias_initializer

        @property
        def kernel_regularizer(self):
            return self.cell.kernel_regularizer

        @property
        def recurrent_regularizer(self):
            return self.cell.recurrent_regularizer

        @property
        def bias_regularizer(self):
            return self.cell.bias_regularizer

        @property
        def kernel_constraint(self):
            return self.cell.kernel_constraint

        @property
        def recurrent_constraint(self):
            return self.cell.recurrent_constraint

        @property
        def bias_constraint(self):
            return self.cell.bias_constraint

        @property
        def dropout(self):
            return self.cell.dropout

        @property
        def recurrent_dropout(self):
            return self.cell.recurrent_dropout

        def get_config(self):
            config = {'units': self.units,
                      'activation': activations.serialize(self.activation),
                      'use_bias': self.use_bias,
                      'kernel_initializer':
                          initializers.serialize(self.kernel_initializer),
                      'recurrent_initializer':
                          initializers.serialize(self.recurrent_initializer),
                      'bias_initializer': initializers.serialize(self.bias_initializer),
                      'kernel_regularizer':
                          regularizers.serialize(self.kernel_regularizer),
                      'recurrent_regularizer':
                          regularizers.serialize(self.recurrent_regularizer),
                      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                      'activity_regularizer':
                          regularizers.serialize(self.activity_regularizer),
                      'kernel_constraint': constraints.serialize(self.kernel_constraint),
                      'recurrent_constraint':
                          constraints.serialize(self.recurrent_constraint),
                      'bias_constraint': constraints.serialize(self.bias_constraint),
                      'dropout': self.dropout,
                      'recurrent_dropout': self.recurrent_dropout}
            base_config = super(CGRU, self).get_config()
            del base_config['cell']
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            if 'implementation' in config:
                config.pop('implementation')
            return cls(**config)


class CLSTMCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(CLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (2 * self.units, 2 * self.units)
        self.output_size = 2 * self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        self.real_kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                           name='real_kernel',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.imaginary_kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                                name='imaginary_kernel',
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        self.real_recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                                     name='real_recurrent_kernel',
                                                     initializer=self.recurrent_initializer,
                                                     regularizer=self.recurrent_regularizer,
                                                     constraint=self.recurrent_constraint)
        self.imaginary_recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                                          name='imaginary_recurrent_kernel',
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.real_bias = self.add_weight(shape=(self.units * 4,),
                                             name='real_bias',
                                             initializer=bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.imaginary_bias = self.add_weight(shape=(self.units * 4,),
                                                  name='imaginary_bias',
                                                  initializer=bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
        else:
            self.real_bias = None
            self.imaginary_bias = None
        self.real_kernel_i = self.real_kernel[:, :self.units]
        self.imaginary_kernel_i = self.imaginary_kernel[:, :self.units]
        self.cat_kernel_i = K.concatenate(
            [K.concatenate([self.real_kernel_i, -self.imaginary_kernel_i], axis=-1),
             K.concatenate([self.imaginary_kernel_i, self.real_kernel_i], axis=-1)],
            axis=0
        )
        self.real_kernel_f = self.real_kernel[:, self.units: self.units * 2]
        self.imaginary_kernel_f = self.imaginary_kernel[:, self.units: self.units * 2]
        self.cat_kernel_f = K.concatenate(
            [K.concatenate([self.real_kernel_f, -self.imaginary_kernel_f], axis=-1),
             K.concatenate([self.imaginary_kernel_f, self.real_kernel_f], axis=-1)],
            axis=0
        )
        self.real_kernel_c = self.real_kernel[:, self.units * 2: self.units * 3]
        self.imaginary_kernel_c = self.imaginary_kernel[:, self.units * 2: self.units * 3]
        self.cat_kernel_c = K.concatenate(
            [K.concatenate([self.real_kernel_c, -self.imaginary_kernel_c], axis=-1),
             K.concatenate([self.imaginary_kernel_c, self.real_kernel_c], axis=-1)],
            axis=0
        )
        self.real_kernel_o = self.real_kernel[:, self.units * 3:]
        self.imaginary_kernel_o = self.imaginary_kernel[:, self.units * 3:]
        self.cat_kernel_o = K.concatenate(
            [K.concatenate([self.real_kernel_o, -self.imaginary_kernel_o], axis=-1),
             K.concatenate([self.imaginary_kernel_o, self.real_kernel_o], axis=-1)],
            axis=0
        )

        self.real_recurrent_kernel_i = self.real_recurrent_kernel[:, :self.units]
        self.imaginary_recurrent_kernel_i = self.imaginary_recurrent_kernel[:, :self.units]
        self.cat_recurrent_kernel_i = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_i, -self.imaginary_recurrent_kernel_i], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_i, self.real_recurrent_kernel_i], axis=-1)],
            axis=0
        )

        self.real_recurrent_kernel_f = self.real_recurrent_kernel[:, self.units: self.units * 2]
        self.imaginary_recurrent_kernel_f = self.imaginary_recurrent_kernel[:, self.units: self.units * 2]
        self.cat_recurrent_kernel_f = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_f, -self.imaginary_recurrent_kernel_f], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_f, self.real_recurrent_kernel_f], axis=-1)],
            axis=0
        )
        self.real_recurrent_kernel_c = self.real_recurrent_kernel[:, self.units * 2: self.units * 3]
        self.imaginary_recurrent_kernel_c = self.imaginary_recurrent_kernel[:, self.units * 2: self.units * 3]
        self.cat_recurrent_kernel_c = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_c, -self.imaginary_recurrent_kernel_c], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_c, self.real_recurrent_kernel_c], axis=-1)],
            axis=0
        )
        self.real_recurrent_kernel_o = self.real_recurrent_kernel[:, self.units * 3:]
        self.imaginary_recurrent_kernel_o = self.imaginary_recurrent_kernel[:, self.units * 3:]
        self.cat_recurrent_kernel_o = K.concatenate(
            [K.concatenate([self.real_recurrent_kernel_o, -self.imaginary_recurrent_kernel_o], axis=-1),
             K.concatenate([self.imaginary_recurrent_kernel_o, self.real_recurrent_kernel_o], axis=-1)],
            axis=0
        )

        if self.use_bias:
            self.real_bias_i = self.real_bias[:self.units]
            self.imaginary_bias_i = self.imaginary_bias[:self.units]
            self.cat_bias_i = K.concatenate([self.real_bias_i, self.imaginary_bias_i],
                                       axis=0)
            self.real_bias_f = self.real_bias[self.units: self.units * 2]
            self.imaginary_bias_f = self.imaginary_bias[self.units: self.units * 2]
            self.cat_bias_f = K.concatenate([self.real_bias_f, self.imaginary_bias_f],
                                       axis=0)
            self.real_bias_c = self.real_bias[self.units * 2: self.units * 3]
            self.imaginary_bias_c = self.imaginary_bias[self.units * 2: self.units * 3]
            self.cat_bias_c = K.concatenate([self.real_bias_c, self.imaginary_bias_c],
                                       axis=0)
            self.real_bias_o = self.real_bias[self.units * 3:]
            self.imaginary_bias_o = self.imaginary_bias[self.units * 3:]
            self.cat_bias_o = K.concatenate([self.real_bias_o, self.imaginary_bias_o],
                                       axis=0)
        else:
            self.real_bias_i = None
            self.imaginary_bias_i = None
            self.cat_bias_i = None

            self.real_bias_f = None
            self.imaginary_bias_f = None
            self.cat_bias_f = None

            self.real_bias_c = None
            self.imaginary_bias_c = None
            self.cat_bias_c = None

            self.real_bias_o = None
            self.imaginary_bias_o = None
            self.cat_bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):

        # Dropout is a bit tricky here: we really only want to create a mask for half of the input and then duplicate and concatenate it,
        # because of how complex numbers are represented here. In other words, we make a mask for the real component and then
        # duplicate it for the imaginary component
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            input_size = K.int_shape(inputs)[-1] // 2
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs[:, :input_size]),
                self.dropout,
                training=training,
                count=4)
            self._dropout_mask = [K.concatenate([mask, mask]) for mask in self._dropout_mask]
        if (0 < self.recurrent_dropout < 1 and
            self._recurrent_dropout_mask is None):
            state_size = K.int_shape(states[0])[-1] // 2
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0][:, :state_size]),
                self.recurrent_dropout,
                training=training,
                count=4
            )
            self._recurrent_dropout_mask = [K.concatenate([mask, mask]) for mask in self._recurrent_dropout_mask]


        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]
        c_tm1 = states[1]

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            x_i = K.dot(inputs_i, self.cat_kernel_i)
            x_f = K.dot(inputs_f, self.cat_kernel_f)
            x_c = K.dot(inputs_c, self.cat_kernel_c)
            x_o = K.dot(inputs_o, self.cat_kernel_o)
            if self.use_bias:

                x_i = K.bias_add(x_i, self.cat_bias_i)

                x_f = K.bias_add(x_f, self.cat_bias_f)

                x_c = K.bias_add(x_c, self.cat_bias_c)

                x_o = K.bias_add(x_o, self.cat_bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1

            i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.cat_recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.cat_recurrent_kernel_f))
            c = c_elem_mult(f, c_tm1, self.units) \
                + c_elem_mult(i, self.activation(x_c + K.dot(h_tm1_c,
                                                             self.cat_recurrent_kernel_c)), self.units)
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.cat_recurrent_kernel_o))
        else:
            # TODO: implement the second version as seen in keras
            None
        h = c_elem_mult(o, self.activation(c), self.units)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, c]
    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(CLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CLSTM(RNN):
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
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 2:
            warnings.warn('`implementation=2 has not been implemented.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = CLSTMCell(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
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
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        super(CLSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        def call(self, inputs, mask=None, training=None, initial_state=None):
            self.cell._dropout_mask = None
            self.cell._recurrent_dropout_mask = None
            return super(CLSTM, self).call(inputs,
                                                mask=mask,
                                                training=training,
                                                initial_state=initial_state)

        @property
        def units(self):
            return self.cell.units

        @property
        def activation(self):
            return self.cell.activation

        @property
        def use_bias(self):
            return self.cell.use_bias

        @property
        def kernel_initializer(self):
            return self.cell.kernel_initializer

        @property
        def recurrent_initializer(self):
            return self.cell.recurrent_initializer

        @property
        def bias_initializer(self):
            return self.cell.bias_initializer

        @property
        def kernel_regularizer(self):
            return self.cell.kernel_regularizer

        @property
        def recurrent_regularizer(self):
            return self.cell.recurrent_regularizer

        @property
        def bias_regularizer(self):
            return self.cell.bias_regularizer

        @property
        def kernel_constraint(self):
            return self.cell.kernel_constraint

        @property
        def recurrent_constraint(self):
            return self.cell.recurrent_constraint

        @property
        def bias_constraint(self):
            return self.cell.bias_constraint

        @property
        def dropout(self):
            return self.cell.dropout

        @property
        def recurrent_dropout(self):
            return self.cell.recurrent_dropout

        def get_config(self):
            config = {'units': self.units,
                      'activation': activations.serialize(self.activation),
                      'use_bias': self.use_bias,
                      'kernel_initializer':
                          initializers.serialize(self.kernel_initializer),
                      'recurrent_initializer':
                          initializers.serialize(self.recurrent_initializer),
                      'bias_initializer': initializers.serialize(self.bias_initializer),
                      'kernel_regularizer':
                          regularizers.serialize(self.kernel_regularizer),
                      'recurrent_regularizer':
                          regularizers.serialize(self.recurrent_regularizer),
                      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                      'activity_regularizer':
                          regularizers.serialize(self.activity_regularizer),
                      'kernel_constraint': constraints.serialize(self.kernel_constraint),
                      'recurrent_constraint':
                          constraints.serialize(self.recurrent_constraint),
                      'bias_constraint': constraints.serialize(self.bias_constraint),
                      'dropout': self.dropout,
                      'recurrent_dropout': self.recurrent_dropout}
            base_config = super(CLSTM, self).get_config()
            del base_config['cell']
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            if 'implementation' in config:
                config.pop('implementation')
            return cls(**config)


