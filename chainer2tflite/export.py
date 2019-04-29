# -*- coding: utf-8 -*-
import collections
import heapq

# logging
import logging
from logging import getLogger

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from chainer import function
from chainer import function_node

from chainer import serializers
from chainer import Link, Chain, ChainList

import flatbuffers

# FIXME(LTE): Find better way of importing tflite
from . import tflite
from .tflite import Buffer
from .tflite import TensorType
from .tflite import Tensor
from .tflite import Model
from .tflite import OperatorCode
from .tflite import SubGraph

from . import serialize_ops

# default log format
default_fmt = logging.Formatter(
    '[%(asctime)s] %(levelname)s '
    '(%(process)d) %(name)s : %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')

# set up handler
try:
    # Rainbow Logging
    import sys
    from rainbow_logging_handler import RainbowLoggingHandler
    default_handler = RainbowLoggingHandler(sys.stdout)
except Exception:
    default_handler = logging.StreamHandler()

default_handler.setFormatter(default_fmt)
default_handler.setLevel(logging.INFO)

logger = getLogger(__name__)
logger.addHandler(default_handler)

_function_types = (function.Function, function_node.FunctionNode)

#
# TODO(LTE):
#
# * [ ] Consider endianness.
# * [ ] Label input/output tensor before serializing Tensor
#

# Based on Chainre's caffe exporter

# ===========================================================================
#
# Copyright (c) 2015 Preferred Infrastructure, Inc.
#
# Copyright (c) 2015 Preferred Networks, Inc.
#
# See LICENSE of Chainer for details.


def _dump_graph(outputs):
    fan_out = collections.defaultdict(int)
    cand_funcs = []

    def add_cand_to_check(cands):
        for cand in cands:
            x = cand.creator
            if x is None:
                continue
            if x not in fan_out:
                # `len(fan_out)` is in order to avoid comparing `x`
                heapq.heappush(cand_funcs, (-x.rank, len(fan_out), x))
            fan_out[x] += 1

    add_cand_to_check(outputs)
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        assert isinstance(func, _function_types)
        add_cand_to_check(func.inputs)

    ret = []
    cand_funcs = []
    seen_set = set()

    def add_cand(cands):
        cands = [cand.creator for cand in cands if cand.creator is not None]
        for x in cands:
            if x in seen_set:
                continue
            order = 1
            if fan_out[x] == 1 and len(cands) == 1:
                order = -len(seen_set)
            # Negate since heapq is min-heap
            # `len(seen_set)` is in order to avoid comparing `x`
            heapq.heappush(cand_funcs, (order, -x.rank, -len(seen_set), x))
            seen_set.add(x)

    add_cand(outputs)
    while cand_funcs:
        _, _, _, func = heapq.heappop(cand_funcs)
        ret.append(func)
        add_cand(func.inputs)

    return ret[::-1]


# ===========================================================================


class TensorFlowLiteSerializer:
    def __init__(self):
        self.builder = flatbuffers.Builder(0)

        self.buffers = []  # Records Buffer pos

        # 0th buffer must have empty buffer
        self.SerializeBuffer(None)

        self.tensors = []  # Records Tensor pos

        # List of builtin opcodes.
        # This information is required for serializing Model
        self.builtin_opcodes = []

        # List of network operators
        self.operators = []

        # The number of tensor ids(for inputs/outputs in subgraph)
        self.num_tensor_ids = 0

        # connection <-> tensor id map
        self.conn_to_tensor_id = {}

        self.logger = logger

    def FindConnection(self, conn_name):
        if conn_name in self.conn_to_tensor_id:
            return self.conn_to_tensor_id[conn_name]

        return None

    def RegisterConnection(self, conn_name, tensor_id):
        if conn_name in self.conn_to_tensor_id:
            logger.fatal('{} is already registered.'.format(conn_name))
            raise

        self.conn_to_tensor_id[conn_name] = tensor_id

    def EmitTensorId(self):

        # Sequentially assign connection id.
        tensor_id = self.num_tensor_ids

        self.num_tensor_ids = self.num_tensor_ids + 1

        return tensor_id

    def RegisterBuiltinOpcode(self, opcode):
        """Register tflite's Builtin opcode

        Args:

            opcode (tflite enum) : BuiltinOpcode enum

        Returns:

            An arary index to registered opcode
        """

        if opcode in self.builtin_opcodes:
            # opcode is already registered
            return self.builtin_opcodes.index(opcode)

        # Add opcode
        self.builtin_opcodes.append(opcode)

        return self.builtin_opcodes.index(opcode)

    def SerializeBuffer(self, data):
        """
            data : bytearray or None(empty)
        """

        data_len = 0
        if data is not None:
            data_len = len(data)

        if data_len > 0:
            # Serialize tensor data: [uint8]
            # https://github.com/google/flatbuffers/issues/4668
            buffer_start = tflite.Buffer.BufferStartDataVector(
                self.builder, data_len)

            # We need to seek the header to correct place before writing into
            # Bytes array
            self.builder.head = self.builder.head - data_len
            self.builder.Bytes[self.builder.head:(self.builder.head +
                                                  data_len)] = data

            tf_data = self.builder.EndVector(data_len)

        tflite.Buffer.BufferStart(self.builder)
        if data_len > 0:
            tflite.Buffer.BufferAddData(self.builder, tf_data)
        tf_buffer = tflite.Buffer.BufferEnd(self.builder)

        buffer_id = len(self.buffers)
        self.buffers.append(tf_buffer)

        return (buffer_id, tf_buffer)

    def SerializeTensor(self, name, dtype, shape, data):
        """Serialize Tensor.

        Currently we only support Tensor with float32 format.

        Args:
            name (string): (Unique) name of Tensor.
            dtype (numpy.dtype): Tensor data type.
            shape ([int]): Tensor shape information.
            data (chainer.Variable or numpy.ndarray): Tensor data.
                Create empty tensor when `data` is None

        Returns:
            tensor id(int)
        """

        # TODO(LTE): Support other types
        tf_type = None
        if dtype == 'float32':
            tf_type = tflite.TensorType.TensorType.FLOAT32
        elif dtype == 'int32':
            tf_type = tflite.TensorType.TensorType.INT32
        else:
            print('Unsupported data type :', dtype)
            raise


        # Serialize Tensor data: [ubyte]
        if data is not None:
            # Sequentially assign buffer id.
            buffer_id = len(self.buffers)

            tensor_values = data.flatten()  # numpy.ndarray
            self.SerializeBuffer(tensor_values.tobytes())
        else:
            # point to empty buffer
            buffer_id = 0

        # Serialize shape: [int32]
        tflite.Tensor.TensorStartShapeVector(self.builder, len(shape))

        for i in reversed(range(len(shape))):
            self.builder.PrependInt32(shape[i])
        tf_shape = self.builder.EndVector(len(shape))

        tf_name = self.builder.CreateString(name)

        # Buld Tensor table
        tflite.Tensor.TensorStart(self.builder)
        tflite.Tensor.TensorAddName(self.builder, tf_name)
        tflite.Tensor.TensorAddShape(self.builder, tf_shape)
        tflite.Tensor.TensorAddType(self.builder, tf_type)
        tflite.Tensor.TensorAddBuffer(self.builder, buffer_id)
        tf_tensor = tflite.Tensor.TensorEnd(self.builder)

        tensor_id = self.EmitTensorId()
        self.tensors.append(tf_tensor)

        return tensor_id


    def SerializeSubGraph(self, inputs, outputs):
        """Serialize SubGraph.

        Args:
            inputs ([int]) : List of input ids.
            outputs ([int]) : List of output ids.
        """

        logger.info("Num inputs = %d", len(inputs))
        logger.info("  %s", inputs)
        logger.info("Num outputs = {}".format(len(outputs)))
        logger.info("  %s", outputs)
        logger.info("Num tensors = {}".format(len(self.tensors)))
        logger.info("Num operators = {}".format(len(self.operators)))

        # [Inputs]
        tflite.SubGraph.SubGraphStartInputsVector(self.builder, len(inputs))
        for i in reversed(inputs):
            self.builder.PrependInt32(i)
        tf_inputs = self.builder.EndVector(len(inputs))

        # [Outputs]
        tflite.SubGraph.SubGraphStartOutputsVector(self.builder, len(outputs))
        for o in reversed(outputs):
            self.builder.PrependInt32(o)
        tf_outputs = self.builder.EndVector(len(inputs))

        # [Operators]
        tflite.SubGraph.SubGraphStartOperatorsVector(self.builder,
                                                     len(self.operators))
        for o in reversed(self.operators):
            self.builder.PrependUOffsetTRelative(o)
        tf_operators = self.builder.EndVector(len(self.operators))

        # [Tensors]
        logger.info('self.tensors = %d', len(self.tensors))
        tflite.SubGraph.SubGraphStartTensorsVector(self.builder,
                                                   len(self.tensors))
        for tensor_pos in reversed(self.tensors):
            logger.info('tensor_pos = %d', tensor_pos)
            self.builder.PrependUOffsetTRelative(tensor_pos)
        tf_tensors = self.builder.EndVector(len(self.tensors))

        # TODO(syoyo): subgraph name
        tf_name = self.builder.CreateString("Nyaan")

        tflite.SubGraph.SubGraphStart(self.builder)
        tflite.SubGraph.SubGraphAddInputs(self.builder, tf_inputs)
        tflite.SubGraph.SubGraphAddOutputs(self.builder, tf_outputs)
        tflite.SubGraph.SubGraphAddOperators(self.builder, tf_operators)
        tflite.SubGraph.SubGraphAddTensors(self.builder, tf_tensors)
        tflite.SubGraph.SubGraphAddName(self.builder, tf_name)

        subgraph = tflite.SubGraph.SubGraphEnd(self.builder)

        return subgraph

    def SerializeModel(self, subgraph):

        # [Buffers]
        tflite.Model.ModelStartBuffersVector(self.builder, len(self.buffers))

        for i in reversed(range(len(self.buffers))):
            self.builder.PrependUOffsetTRelative(self.buffers[i])

        tf_buffers = self.builder.EndVector(len(self.buffers))

        # [Subgraphs]
        # Currently we only support 1 subgraphs in a model.
        tflite.Model.ModelStartSubgraphsVector(self.builder, 1)
        self.builder.PrependUOffsetTRelative(subgraph)
        tf_subgraphs = self.builder.EndVector(1)

        # [OperatorCodes]
        tf_opcodes = []
        for k in self.builtin_opcodes:
            tflite.OperatorCode.OperatorCodeStart(self.builder)
            logger.info('code = %d', k)
            tflite.OperatorCode.OperatorCodeAddBuiltinCode(self.builder, k)
            tf_opcode = tflite.OperatorCode.OperatorCodeEnd(self.builder)
            logger.info('tf_opcode = %d', tf_opcode)

            tf_opcodes.append(tf_opcode)

        tflite.Model.ModelStartOperatorCodesVector(self.builder,
                                                   len(tf_opcodes))
        for i in reversed(range(len(tf_opcodes))):
            self.builder.PrependUOffsetTRelative(tf_opcodes[i])
        opcodes = self.builder.EndVector(len(tf_opcodes))

        tflite.Model.ModelStart(self.builder)

        # version must be 3(or higher?)
        tflite.Model.ModelAddVersion(self.builder, 3)
        tflite.Model.ModelAddSubgraphs(self.builder, tf_subgraphs)
        tflite.Model.ModelAddBuffers(self.builder, tf_buffers)
        tflite.Model.ModelAddOperatorCodes(self.builder, opcodes)
        model = tflite.Model.ModelEnd(self.builder)

        return model

    def GetOutput(self, rootTable):

        # file_identifier is missing in python binding
        # (At least flatbuffers ~1.11).
        # https://github.com/google/flatbuffers/issues/4814
        #
        # `file_identifier` is required when reading it in TensorFlow Lite C++.
        # Manually add `file_identifier` here

        file_identifier = 'TFL3'

        prepSize = flatbuffers.number_types.UOffsetTFlags.bytewidth + len(
            file_identifier)  # = 8
        self.builder.Prep(self.builder.minalign, prepSize)

        b = bytes(file_identifier, encoding='utf-8')
        for i in reversed(b):
            self.builder.PrependByte(i)

        self.builder.Finish(rootTable)

        return self.builder.Output()


class TensorFlowLiteConverter(object):

    debug = False

    def __init__(self, tflitemodel=None):
        self.tflitemodel = tflitemodel
        # key:string, val:dict(key: func, val: index)
        self.naming_map = collections.defaultdict(dict)

        # Placeholder input tensor id
        # Will be found during `dump_function_object`
        self.inputs = {}

        # List of input names
        self.input_names = []

    def _get_layer_name(self, layer):
        """Generate layer name like "Convolution2DFunction-10-2".

        The first number means rank of the layer (depth from the top),
        and the second number is for preventing duplication
        (different layer objects can have same rank)

        Args:
            layer (~chainer.Function_node): Function object
        Returns:
            str: A string to be used for the ``name`` field of the graph
                in the exported Caffe model.

        """
        label = '{}-{}'.format(layer.label, layer.rank)
        d = self.naming_map[label]
        if layer not in d.keys():
            d[layer] = len(d) + 1
        return '{}-{}'.format(label, d[layer])

    def _get_parent_name(self, parent_):
        if parent_ is None:
            return 'data'
        return self._get_layer_name(parent_)

    def dump_function_object(self, func, tf_serializer):

        assert isinstance(func, _function_types)
        layer_name = self._get_layer_name(func)

        parent_layer_names = [
            self._get_parent_name(input_.creator) for input_ in func.inputs
        ]

        layer = None

        for input_ in func.inputs:
            logger.info('input name = %s', input_.name)

        logger.info('label = %s', func.label)
        logger.info('len(inputs) = %d', len(func.inputs))
        logger.info('top = %s', layer_name)
        logger.info('parent_layer_names = %s', parent_layer_names)

        # NOTE(LTE): `func.outputs` is a weakref.
        # So use '()' to deref it when you access `func.outputs`

        if func.label == 'LinearFunction':
            #
            # TODO(syoyo): Convert LinearFunction + ReLU to
            # FULLY_CONNECTED with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('Linear in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            b = None
            if len(func.inputs) == 2:
                I, W = func.inputs
            else:  # guess 3
                I, W, b = func.inputs

            # input
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, I.shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.dtype, I.shape, I.data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # W
            W_id = tf_serializer.SerializeTensor(parent_layer_names[1],
                                                 W.dtype, W.shape, W.data)

            # b
            b_id = -1  # -1 = optional
            if b is not None:
                if b.data is not None:
                    b_id = tf_serializer.SerializeTensor(
                        parent_layer_names[2], b.dtype, b.shape, b.data)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype, _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            activation_function = 'NONE'
            serialize_ops.SerializeOpFullyConnected(tf_serializer, activation_function,
                                                    input_id, output_id, W_id,
                                                    b_id)

        elif func.label == 'AveragePooling2D':
            #
            # TODO(syoyo): Convert AveragePooling2D + ReLU to
            # AVERAGE_POOL_2D with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('AveragePooling2D in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) == 1

            I = func.inputs[0]

            in_shape = I.shape
            in_data = I.data
            if len(in_shape) == 4:
                # Assume NCHW
                # Apply NHWC conversion
                in_shape = (I.shape[0], I.shape[2], I.shape[3], I.shape[1])
                if in_data is not None:
                    in_data = np.transpose(I.data, (0, 2, 3, 1))

            # input
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, in_shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.dtype, in_shape, in_data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise


            # output
            _output = func.outputs[0]

            output_shape = _output().shape

            if len(output_shape) == 4:
                # NCHW -> NHWC
                output_shape = (
                            _output().shape[0],
                            _output().shape[2],
                            _output().shape[3],
                            _output().shape[1])

            #print("average_pool_2d.output.shape = {}".format(output_shape))
            logger.info("average_pool_2d.output.shape = {}".format(output_shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype,output_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            # options

            # Padding must be same for both axis.
            assert func.ph == func.pw

            # Padding must be 0 or 1
            assert func.ph == 0 or func.ph == 1

            activation_function = 'NONE'

            padding = 'VALID' if func.ph == 0 else 'SAME'
            stride = [func.sx, func.sy]
            filter_size = [func.kw, func.kh]
            serialize_ops.SerializeAveragePooling2D(tf_serializer,
                                                    input_id, output_id, activation_function,
                                                    padding, stride, filter_size)

        elif func.label == 'MaxPooling2D':
            #
            # TODO(syoyo): Convert MaxPooling2D + ReLU to
            # MAX_POOL_2D with ReLU as fused_activation_fuction
            #
            for _input in func.inputs:
                logger.info('MaxPooling2D in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            assert len(func.inputs) == 1

            I = func.inputs[0]

            in_shape = I.shape
            in_data = I.data
            if len(in_shape) == 4:
                # Assume NCHW
                # Apply NHWC conversion
                in_shape = (I.shape[0], I.shape[2], I.shape[3], I.shape[1])
                if in_data is not None:
                    in_data = np.transpose(I.data, (0, 2, 3, 1))

            # input
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, in_shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.dtype, in_shape, in_data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise


            # output
            _output = func.outputs[0]

            output_shape = _output().shape

            if len(output_shape) == 4:
                # NCHW -> NHWC
                output_shape = (
                            _output().shape[0],
                            _output().shape[2],
                            _output().shape[3],
                            _output().shape[1])

            #print("average_pool_2d.output.shape = {}".format(output_shape))
            logger.info("max_pool_2d.output.shape = {}".format(output_shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype,output_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            # options

            # Padding must be same for both axis.
            assert func.ph == func.pw

            # Padding must be 0 or 1
            assert func.ph == 0 or func.ph == 1

            activation_function = 'NONE'

            padding = 'VALID' if func.ph == 0 else 'SAME'
            stride = [func.sx, func.sy]
            filter_size = [func.kw, func.kh]
            serialize_ops.SerializeMaxPooling2D(tf_serializer,
                                                    input_id, output_id, activation_function,
                                                    padding, stride, filter_size)
        elif func.label == 'ReLU':

            assert (len(func.inputs) == 1)

            # TODO(LTE): Support non-float32 type

            # input
            I = func.inputs[0]
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, 'float32', I.shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                layer_name + '_input0', 'float32', I.shape, I.data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      'float32', _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            serialize_ops.SerializeOpReLU(tf_serializer, input_id, output_id)

        elif func.label == 'Pad':

            assert (len(func.inputs) == 1)
            assert func.mode == 'constant'

            print(func.pad_bw)

            values = float(0.0)
            if 'constant_values' in func.keywords:
                values = func.keywords['constant_values']
                if not isinstance(values, int) and len(values) > 1:
                    raise ValueError(
                        'tflite doesn\'t support multiple constant values for Pad '
                        'operation')
                elif not isinstance(values, int):
                    values = float(values[0])
                else:
                    values = float(values)

            if (values > 1e-5) or (values < -1e-5):
                raise ValueError(
                    'tflite doesn\'t support padding constant value other than zero')


            new_shape = func.outputs[0]().shape
            print('pad new_shape = ', new_shape)

            I = func.inputs[0]
            in_shape = I.shape
            in_data = I.data

            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, in_shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', in_shape, in_data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # create a constant value tensor for padding.

            # tflite = 2D tensor with [begin, end]x ndim. For example:
            # [[pad0_b, pad0_e],
            #  [pad1_b, pad1_e],
            #  [pad2_b, pad2_e],
            #  [pad3_b, pad3_e]]
            print('func_pad = ', func.pad_bw)
            padding_values = []
            for pad_bw in func.pad_bw:
                if isinstance(pad_bw, np.ndarray):
                    padding_values.append([pad_bw[0], pad_bw[1]])
                else:
                    padding_values.append([pad_bw, pad_bw])

            print(padding_values)

            # paddig tensor must have same array length for the first axis with input tensor
            padding = np.array(padding_values, np.int32)

            print('padding.shape = ', padding.shape)
            print('padding = ', padding)
            padding_id = tf_serializer.SerializeTensor(
                layer_name + '_padding', I.dtype, padding.shape, padding)

            # output
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype, new_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)


            serialize_ops.SerializeOpPad(tf_serializer,
                input_id, output_id, padding_id)


        elif func.label == 'Reshape':

            assert (len(func.inputs) == 1)

            new_shape = func.outputs[0]().shape

            logger.info('Reshape : {}'.format(new_shape))

            I = func.inputs[0]
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, I.shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.dtype, I.shape, I.data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype, new_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)
            serialize_ops.SerializeOpReshape(tf_serializer,
                input_id, output_id, new_shape)

        elif func.label == 'ResizeImages':

            # Assume float32 image
            # TODO(LTE): Support multiple channels

            for _input in func.inputs:
                logger.info('ResizeImages in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            print('resize_images', func.inputs)
            print('output_shape', func.out_H, func.out_W)

            I = func.inputs[0]

            in_shape = I.shape
            in_data = I.data

            if len(in_shape) == 4:
                # Assume NCHW
                # Apply NHWC conversion
                in_shape = (I.shape[0], I.shape[2], I.shape[3], I.shape[1])
                if in_data is not None:
                    in_data = np.transpose(I.data, (0, 2, 3, 1))

            # input
            if I.name in self.input_names:
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.dtype, in_shape, None)
                self.inputs[I.name] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', 'float32', in_shape, in_data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # new_shape(1D tensor with 2 elements)
            new_shape_name = layer_name + '_new_shape'
            new_shape_id = tf_serializer.SerializeTensor(new_shape_name,
                                                 'int32', [2], np.array([func.out_H, func.out_W], dtype=np.int32))

            # output
            _output = func.outputs[0]
            output_shape = _output().shape

            if len(output_shape) == 4:
                # NCHW -> NHWC
                output_shape = (_output().shape[0],
                                _output().shape[2],
                                _output().shape[3],
                                _output().shape[1])

            logger.info("output.shape = {}".format(output_shape))
            print('len(shape) = {}'.format(len(output_shape)))
            print('ty(shape) = {}'.format(type(output_shape)))
            print('resize_images.out_shape = {}'.format(output_shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      I.dtype, output_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            serialize_ops.SerializeOpResizeImages(tf_serializer, input_id, output_id, new_shape_id)

        elif func.label == '_ + _':
            # Add

            if len(func.inputs) != 2:
                logger.fatal('The number of inputs for `Add` op must be two(2) but got got {}'.format(len(func.inputs)))

            for _input in func.inputs:
                logger.info('Add in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            input_ids = []

            for (i, inp) in enumerate(func.inputs):
                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    # Constant
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input'.format(i), inp.data.dtype, inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindConnection(parent_layer_names[i])
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name,
                                                      func.inputs[0].dtype, _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            serialize_ops.SerializeOpAdd(tf_serializer, input_ids[0], input_ids[1], output_id)

        elif func.label == '_ - _':
            # Sub

            if len(func.inputs) != 2:
                logger.fatal('The number of inputs for `Sub` op must be two(2) but got got {}'.format(len(func.inputs)))

            for _input in func.inputs:
                logger.info('Sub in %s(id %d)',
                            self._get_parent_name(_input), id(_input))

            input_ids = []

            for (i, inp) in enumerate(func.inputs):
                # input
                if inp.name in self.input_names:
                    # Placeholder input
                    input_id = tf_serializer.SerializeTensor(
                        inp.name, inp.data.dtype, inp.shape, None)
                    self.inputs[inp.name] = input_id
                elif parent_layer_names[i] == 'data':
                    # Constant
                    input_id = tf_serializer.SerializeTensor(
                        layer_name + '_input'.format(i), inp.data.dtype, inp.shape, inp.data)
                else:
                    input_id = tf_serializer.FindConnection(parent_layer_names[i])
                    # There should have valid connection
                    if input_id is None:
                        logger.fatal('{} not found in connections'.format(
                            parent_layer_names[i]))
                        raise

                input_ids.append(input_id)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))

            output_name = layer_name + '_0'
            output_id = tf_serializer.SerializeTensor(output_name,
                                                      'float32', _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            serialize_ops.SerializeOpSub(tf_serializer, input_ids[0], input_ids[1], output_id)
        else:
            logger.fatal("Unknown or unsupported function/link : %s",
                         func.label)

    def __call__(self, _inputs, outputs):

        # register list of input names
        for _inp in _inputs:
            assert _inp.name is not None
            self.input_names.append(_inp.name)


        logger.info('input names = {}'.format(self.input_names))

        dumped_list = _dump_graph(outputs)
        logger.debug('dumpped_list = %s', dumped_list)
        f = None
        tf_serializer = TensorFlowLiteSerializer()

        logger.info('outputs = %s', outputs[0].label)

        try:
            for i in dumped_list:
                self.dump_function_object(i, tf_serializer)

            # Flattern
            input_ids = [self.inputs[name] for name in self.inputs]

            # TODO(LTE): Find output in more rubust way.
            output_ids = [tf_serializer.num_tensor_ids - 1]

            subgraph = tf_serializer.SerializeSubGraph(input_ids, output_ids)
            tfmodel = tf_serializer.SerializeModel(subgraph)

            buf = tf_serializer.GetOutput(tfmodel)

            tflitemodel_filepath = self.tflitemodel

            if tflitemodel_filepath is None:
                tflitemodel_filepath = 'chainer_model.tflite'

            f = open(tflitemodel_filepath, 'wb')
            f.write(buf)

            logger.info("Wrote a file: {} ({} bytes)".format(
                tflitemodel_filepath, len(buf)))

        finally:
            if f is not None:
                f.close()


def export(model, args, filename):

    # forward eval

    # `inputs` contain chainer.Variable with name assigned.
    inputs = []
    if isinstance(args, tuple):
        args = list(args)

    if isinstance(args, list):
        for i, arg in enumerate(args):
            if isinstance(arg, chainer.get_array_types()):
                input_name = 'input{}'.format(i)
                args[i] = chainer.Variable(arg, name=input_name)
            else:
                assert isinstance(arg, chainer.Variable)
                if args[i].name is None:
                    # assign name
                    args[i].name = 'input0'

            inputs.append(args[i])

        outputs = model(*args)
    elif isinstance(args, chainer.get_array_types()):
        args = chainer.Variable(args, name='input0')
        inputs.append(args)
        outputs = model(args)

    elif isinstance(args, chainer.Variable):
        if args.name is None:
            # assign name
            args.name = 'input0'
        inputs.append(args)
        outputs = model(args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list, tuple, '
            'numpy array, or Chainer Variable. But a {} object was '
            'given.'.format(type(args)))

    assert len(inputs) > 0

    for inp in inputs:
        assert inp.name is not None
        logger.info('DBG: input name = {}'.format(inp.name))

    if isinstance(outputs, variable.Variable):
        outputs = [outputs]
    assert isinstance(outputs, (tuple, list))

    logger.info('# of outpus = {}'.format(len(outputs)))

    for i, outp in enumerate(outputs):
        assert isinstance(outp, variable.Variable)
        if outp.name is None:
            outp.name = 'output{}'.format(i)

        logger.info('output name[{}] = {}'.format(i, outp.name))

    converter = TensorFlowLiteConverter(filename)
    converter(inputs, outputs)

    # Chainer's result
    return inputs, outputs
