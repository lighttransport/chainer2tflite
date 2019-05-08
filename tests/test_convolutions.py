import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

class TestConvolution2D(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():

                    # arg = [in_chan, out_chan, ksize, stride, pad]
                    self.l1 = L.Convolution2D(None, 2, 3, 1, 0)

            def __call__(self, x):
                return self.l1(x)


        self.model = Model()

        self.x = input_generator.increasing(1, 4, 5, 5)

    def test_output(self):
        self.expect(self.model, self.x)

class TestConvolution2DKsize2(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():

                    # arg = [in_chan, out_chan, ksize, stride, pad]
                    self.l1 = L.Convolution2D(None, 2, 2, 1, 0)

            def __call__(self, x):
                return self.l1(x)


        self.model = Model()

        self.x = input_generator.increasing(1, 4, 5, 5)

    def test_output(self):
        self.expect(self.model, self.x)

class TestConvolution2D_pad1(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():

                    # arg = [in_chan, out_chan, ksize, stride, pad]
                    self.l1 = L.Convolution2D(None, 2, 3, 1, 1)

            def __call__(self, x):
                return self.l1(x)


        self.model = Model()

        self.x = input_generator.increasing(1, 4, 5, 5)

    def test_output(self):
        self.expect(self.model, self.x)

