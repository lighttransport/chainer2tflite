"""
  - Create simple ReLU network.
  - Write it to tflite
  - Read tflite model and compare result.
"""
import sys

import logging
logger = logging.getLogger('chainer2tflite')
logger.setLevel(logging.DEBUG)

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import function
from chainer import function_node

from chainer import serializers
from chainer import Link, Chain, ChainList

sys.path.append("../../")
import chainer2tflite

from tensorflow.lite.python import interpreter as interpreter_wrapper

class MyNetwork(Chain):
    def __init__(self):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            pass

    def forward(self, x):
        return F.relu(x)

chainer.print_runtime_info()

model = MyNetwork()

x = chainer.Variable(np.zeros((1, 16), dtype=np.float32))

# for some reason, need to pass the array(or tuple) of variables
filename = "relu.tflite"
chainer2tflite.export(model, [x], filename)

interpreter = interpreter_wrapper.Interpreter(model_path=filename)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
assert input_details[0]['dtype'] == np.float32

print('input', input_details)
print('output', output_details)

input_value = np.random.rand(1, 16).astype(np.float32)
input_value = input_value * 2.0 - 1.0
interpreter.set_tensor(input_details[0]['index'], input_value)

interpreter.invoke()

tfl_result = interpreter.get_tensor(output_details[0]['index'])

y = model(input_value)
cn_result = y.array
assert isinstance(cn_result, np.ndarray)

# compare result
np.testing.assert_allclose(cn_result, tfl_result, rtol=1e-5, atol=1e-5)

# :tada:
print('\U0001F389')
