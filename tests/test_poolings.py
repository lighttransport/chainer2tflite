import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

import pytest

class TestAveragePooling2D(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.average_pooling_2d, [2, 1, 0], None)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestAveragePooling2DPad1(TFLiteModelTest):

    def setUp(self):

        padding = 1
        # arg = [kernel_size, stride, padding]
        self.model = Model(F.average_pooling_2d, [2, 1, padding], None)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestMaxPooling2D(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.max_pooling_2d, [2, 1, 0], False)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestMaxPooling2DPad1(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.max_pooling_2d, [2, 1, 1], False)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestMaxPooling2DCoverall(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.max_pooling_2d, [3, 2, 1], True)
        self.x = input_generator.increasing(1, 3, 12, 12)

    def test_output(self):
        self.expect(self.model, self.x)

# Disable due to different padded value(-inf vs -max)
#class TestMaxPooling2DPad1x2Stride2(TFLiteModelTest):
#
#    def setUp(self):
#
#        # arg = [kernel_size, stride, padding]
#        self.model = Model(F.max_pooling_2d, [2, 2, [1,2]], False)
#        self.x = input_generator.increasing(1, 3, 8, 8)
#
#    def test_output(self):
#        self.expect(self.model, self.x)

class TestMaxPooling2DPad1Stride2(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.max_pooling_2d, [2, 2, 1], False)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestMaxPooling2DPad1Stride2(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        self.model = Model(F.max_pooling_2d, [3, 2, 1], False)
        self.x = input_generator.increasing(1, 128, 240, 320)

    def test_output(self):
        self.expect(self.model, self.x)


class TestUnpooling2D(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        # (stride=None) = use kernel_size
        self.model = Model(F.unpooling_2d, [3, None, 0], False)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)


class TestUnpooling2D_2x(TFLiteModelTest):

    def setUp(self):

        # arg = [kernel_size, stride, padding]
        # (stride=None) = use kernel_size
        self.model = Model(F.unpooling_2d, [2, None, 0], False)
        self.x = input_generator.increasing(1, 3, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class Model(chainer.Chain):

    def __init__(self, ops, args, cover_all):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args
        self.cover_all = cover_all

    def __call__(self, x):
        if self.cover_all is not None:
            return self.ops(*([x] + self.args), cover_all=self.cover_all)
        else:
            return self.ops(*([x] + self.args))
