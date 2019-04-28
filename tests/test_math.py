import chainer
import chainer.functions as F

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest


class TestAdd(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.add = F.add

            def __call__(self, x, y):
                return self.add(x, y)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)
        self.y = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

"""
class TestSub(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, x, y):
                return eval('x - y')

        self.model = Model()
        self.x = chainer.Variable(input_generator.increasing(2, 3))
        self.y = chainer.Variable(input_generator.nonzero_increasing(2, 3) * 0.3)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])
"""
