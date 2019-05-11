import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

class TestDepthwiseConvolution2D(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    # arg = [in_chan, chan_multiplier, ksize, stride, pad]
                    self.l1 = L.DepthwiseConvolution2D(None, 1, 3, 1, 0)

            def __call__(self, x):
                return self.l1(x)


        self.model = Model()

        self.x = input_generator.increasing(1, 4, 5, 5)

    def test_output(self):
        self.expect(self.model, self.x)

class TestDepthwiseConvolution2DMult2(TFLiteModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():

                    # arg = [in_chan, chan_multiplier, ksize, stride, pad]
                    self.l1 = L.DepthwiseConvolution2D(None, 2, 3, 1, 0)

            def __call__(self, x):
                return self.l1(x)


        self.model = Model()

        self.x = input_generator.increasing(1, 4, 5, 5)

    def test_output(self):
        self.expect(self.model, self.x)
