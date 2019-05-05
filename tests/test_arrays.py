import chainer
import chainer.functions as F
import chainer.links as L

from chainer2tflite.testing import input_generator
from tests.helper import TFLiteModelTest

import numpy

import pytest

class TestVStack1D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.vstack)
        self.x = input_generator.increasing(3)
        self.y = input_generator.increasing(3)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestVStack2D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.vstack)
        self.x = input_generator.increasing(2, 3)
        self.y = input_generator.increasing(2, 3)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestVStack3D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.vstack)
        self.x = input_generator.increasing(2, 3, 4)
        self.y = input_generator.increasing(2, 3, 4)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestVStack4D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.vstack)
        self.x = input_generator.increasing(2, 3, 4, 5)
        self.y = input_generator.increasing(2, 3, 4, 5)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestHStack1D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.hstack)
        self.x = input_generator.increasing(3)
        self.y = input_generator.increasing(3)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestHStack2D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.hstack)
        self.x = input_generator.increasing(2, 3)
        self.y = input_generator.increasing(2, 3)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestHStack3D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.hstack)
        self.x = input_generator.increasing(2, 3, 4)
        self.y = input_generator.increasing(2, 3, 4)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestHStack4D(TFLiteModelTest):

    def setUp(self):

        self.model = ModelWithTwoArgs(F.hstack)
        self.x = input_generator.increasing(2, 3, 4, 5)
        self.y = input_generator.increasing(2, 3, 4, 5)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])


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

class TestTranspose(TFLiteModelTest):

    def setUp(self):

        # arg = [axes]
        self.model = Model(F.transpose, [(0, 2, 3, 1)])
        self.x = input_generator.increasing(1, 3, 6, 7)

    def test_output(self):
        self.expect(self.model, self.x)

class TestExpandDims(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = Model(F.expand_dims, [0])
        self.x = input_generator.increasing(1, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class TestExpandDims1(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = Model(F.expand_dims, [1])
        self.x = input_generator.increasing(1, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class TestExpandDims2(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = Model(F.expand_dims, [2])
        self.x = input_generator.increasing(1, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class TestTile(TFLiteModelTest):

    def setUp(self):

        # arg = [replications]
        self.model = Model(F.tile, [(2, 3)])
        self.x = input_generator.increasing(3, 3)

    def test_output(self):
        self.expect(self.model, self.x)

class TestSqueezeNoAxis(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = Model(F.squeeze, [None])
        self.x = input_generator.increasing(1, 3, 1, 2)

    def test_output(self):
        self.expect(self.model, self.x)

class TestSqueeze(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = Model(F.squeeze, [(2, 4)])
        self.x = input_generator.increasing(1, 3, 1, 2, 1)

    def test_output(self):
        self.expect(self.model, self.x)

class TestConcat(TFLiteModelTest):

    def setUp(self):

        # arg = [axis]
        self.model = ModelWithTwoArgs(F.concat)
        self.x = input_generator.increasing(1, 3, 4)
        self.y = input_generator.increasing(1, 3, 4)

    def test_output(self):
        self.expect(self.model, [self.x, self.y])

class TestSpaceToDepth(TFLiteModelTest):

    def setUp(self):

        # arg = [downscaling_factor]
        self.model = Model(F.space2depth, [2])
        self.x = input_generator.increasing(1, 12, 6, 6)

    def test_output(self):
        self.expect(self.model, self.x)

class TestCast(TFLiteModelTest):

    def setUp(self):

        # arg = [typ]
        self.model = Model(F.cast, [numpy.int32])
        self.x = input_generator.increasing(1, 6)

    def test_output(self):
        self.expect(self.model, self.x)


# TODO(LTE): Split requires multiple output support in the converter.
#class TestSplitAxis(TFLiteModelTest):
#
#    def setUp(self):
#
#        # arg = [indices_or_selections, axis]
#        self.model = Model(F.split_axis, [[2], 1])
#        self.x = input_generator.increasing(1, 6)
#
#    def test_output(self):
#        self.expect(self.model, self.x)


class ModelWithTwoArgs(chainer.Chain):

    def __init__(self, ops):
        super(ModelWithTwoArgs, self).__init__()
        self.ops = ops

    def __call__(self, x, y):
        return self.ops([x, y])


class Model(chainer.Chain):

    def __init__(self, ops, args):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args

    def __call__(self, x):
        return self.ops(*([x] + self.args))

