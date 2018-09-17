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
    a = x[:, units]
    b = x[:, units:]
    c = y[:, :units]
    d = y[:, units:]
    real = (a * c) - (b * d)
    imag = (a * d) + (b * c)
    return K.concatenate([real, imag], axis = -1)