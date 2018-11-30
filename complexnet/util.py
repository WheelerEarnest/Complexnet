# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
#
# ------------------------------------------------------------
import numpy as np
import keras.backend as K

def c_elem_mult(x, y, units):
    # Complex element-wise multiplication
    # If we're doing element-wise multiplication with complex numbers, its a bit weird
    # [a,b] * [c,d] = [ac - bd, ad + bc]
    a = x[:, :units]
    b = x[:, units:]
    c = y[:, :units]
    d = y[:, units:]
    real = (a * c) - (b * d)
    imag = (a * d) + (b * c)
    return K.concatenate([real, imag], axis = -1)

def complex_to_float(a):
    """

    :param a: Input expected to be in the shape (units, timesteps)
    :return: float representation, shape (units*2, timesteps)
    """
    # Converts numpy data that is complex64 or complex128 and makes it the float representation
    # so a complex vector would look like [real, imaginary]
    assert a.dtype is np.dtype('complex64') or a.dtype is np.dtype('complex128')
    return np.concatenate((a.real, a.imag), axis=0)

def float_to_complex(a):
    # Converts numpy data that is float32 or float64 to complex representation
    assert a.dtype is np.dtype('float32') or a.dtype is np.dtype('float64')
    shape = np.shape(a)
    assert shape[0] % 2 == 0
    mid = shape[0] // 2
    if a.dtype is np.dtype('float32'):
        c = np.empty((mid, shape[1]), dtype='complex64')
    else:
        c = np.empty((mid, shape[1]), dtype='complex128')
    c.real = a[:mid, :]
    c.imag = a[mid:, :]
    return c