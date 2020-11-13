

#===================================================================
#  This file is subject to the terms and conditions defined in     |
#  file 'LICENSE.txt', which is part of this source code package.  |
#===================================================================

import numpy as np

#================================ sigmoid function f(x) = 1 / (1 + e^(-x)) ==========================
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#================================ Derivative of sigmoid: f'(x) = f(x) * (1 - f(x)) ==========================
def derivSigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

#================================ Mean squared error mean( (y_true - y_pred) ^ 2)  ==========================
def mseLoss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

