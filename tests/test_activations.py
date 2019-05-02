import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

class TestReLU(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.relu = F.relu

            def __call__(self, x):
                return self.relu(x)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)


class TestLeakyReLU(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.leaky_relu = F.leaky_relu

            def __call__(self, x):
                return self.leaky_relu(x)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)

class TestLeakyReLUSlope0__5(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.leaky_relu = F.leaky_relu

            def __call__(self, x):
                # slope = 0.5
                return self.leaky_relu(x, 0.5)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)

class TestSoftmax(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.softmax = F.softmax

            def __call__(self, x):
                return self.softmax(x)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)

