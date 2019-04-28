import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

class TestResizeImages(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                print('resize_ ', x)
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        args = {'output_shape' : (4, 4)}
        self.model = Model(F.resize_images, args, 'x')

        # (batch, channel, height, width) = (1, 1, 2, 2)
        self.x = np.array([[[[64, 32], [64, 32]]]], np.float32)

    def test_output(self):
        self.expect(self.model, self.x)

