# ----------------------------------------------------------
# Author: Wheeler Earnest
#
# Project: Complexnet
#
# ------------------------------------------------------------
import keras as K
import numpy as np
from layers.complex_recurrent import CLSTM

model = K.Sequential()
model.add(K.layers.Input())
