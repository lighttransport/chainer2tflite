import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

class TestPad(TFLiteModelTest):

    def setUp(self):

        # arg = [pad_width, mode]
        # pad with zero constant value.
        self.model = Model(F.pad, [[1, 1], 'constant'])
        self.x = input_generator.increasing(3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class TestPad3D(TFLiteModelTest):

    def setUp(self):

        # arg = [pad_width, mode]
        # No padding for the first axis.
        self.model = Model(F.pad, [[[0, 0], [1, 1], [1, 1]], 'constant'])
        self.x = input_generator.increasing(1, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class Model(chainer.Chain):

    def __init__(self, ops, args):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args

    def __call__(self, x):
        return self.ops(*([x] + self.args))

