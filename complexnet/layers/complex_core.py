# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
#
# ------------------------------------------------------------
import numpy as np
import keras.backend as kb
from keras.layers import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints



class CDense(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        #Input shape: (batch_size, input_dims)
        #Output shape: (batch_size, units)
        #Note: Input vectors should have the real numbers as their first entries
        #   and imaginary numbers after that.
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim',))
        super(CDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        # We have both real and imaginary components so the last dimension must be even
        assert input_shape[-1] % 2 == 0
        # Remember: the input is [real, imaginary], so what looks like a 6 unit input is actually 3
        input_dim = input_shape[-1] // 2
        data_format = kb.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = initializers._compute_fans(kernel_shape, data_format=data_format)

        if self.init_criterion == 'he':
            s = np.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = np.sqrt(1. / (fan_in + fan_out))
        else:
            s = 1.0
        rng = np.random.RandomState(seed=self.seed)

        def init_w_real(shape, dtype=None):
            return rng.normal(loc=0.0,
                              scale=s,
                              size=kernel_shape)
        def init_w_imag(shape, dtype=None):
            return rng.normal(loc=0.0,
                              scale=s,
                              size=kernel_shape)

        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            name='real_kernel',
            shape=kernel_shape,
            initializer=real_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        self.imag_kernel = self.add_weight(
            name='imag_kernel',
            shape=kernel_shape,
            initializer=imag_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        self.kernel = kb.concatenate(
            [kb.concatenate([self.real_kernel, -self.imag_kernel], axis=-1),
             kb.concatenate([self.imag_kernel, self.real_kernel], axis=-1)], axis=0)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs, **kwargs):

        output = kb.dot(inputs, self.kernel)

        if self.use_bias:
            output = kb.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            k_initializer = self.kernel_initializer
        else:
            k_initializer = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': k_initializer,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(CDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))