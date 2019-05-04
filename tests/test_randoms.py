import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

import pytest

@pytest.mark.skip(reason='without using same seed for random, the test fails')
class TestDropout(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.dropout = F.dropout

            def __call__(self, x):
                return self.dropout(x)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)

