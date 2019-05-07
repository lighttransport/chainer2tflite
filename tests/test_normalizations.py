import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

import numpy
import pytest

class TestLocalResponseNormalization(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.op = F.local_response_normalization

            def __call__(self, x):
                return self.op(x, k=1, n=3, alpha=1e-4, beta=0.75)

        self.model = Model()
        self.x = input_generator.increasing(2, 5, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x)


class TestL2Norm(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.op = F.normalize

            def __call__(self, x):

                # Axis must be the last dim of input tensor
                return self.op(x, axis=3)

        self.model = Model()
        self.x = input_generator.increasing(2, 3, 3, 5)

    def test_output(self):
        self.expect(self.model, self.x)

#class TestBatchNormalization(TFLiteModelTest):
#
#    def setUp(self):
#
#        class Model(chainer.Chain):
#
#            def __init__(self):
#                super(Model, self).__init__()
#                with self.init_scope():
#                    kwargs = {}
#                    self.bn = L.BatchNormalization(5, *kwargs)
#
#            def __call__(self, x):
#                return self.bn(x)
#
#        self.model = Model()
#        self.x = input_generator.increasing(2, 5)
#
#    def test_output(self):
#        self.expect(self.model, self.x)

