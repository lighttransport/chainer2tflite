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

import tflite.Buffer
import tflite.TensorType
import tflite.Tensor
import tflite.Model
from tflite import FullyConnectedOptions, Operator, OperatorCode, BuiltinOperator, BuiltinOptions, SubGraph
from tflite import ReshapeOptions

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

    def SerializeTensor(self, name, shape, data):
        """Serialize Tensor.

        Currently we only support Tensor with float32 format.

        Args:
            name (string): (Unique) name of Tensor.
            shape ([int]): Tensor shape information.
            data (chainer.Variable or numpy.ndarray): Tensor data.
                Create empty tensor when `data` is None

        Returns:
            tensor id(int)
        """

        # TODO(LTE): Support other types
        if data is not None:
            assert data.dtype == 'float32'

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
        tflite.Tensor.TensorAddType(self.builder,
                                    tflite.TensorType.TensorType.FLOAT32)
        tflite.Tensor.TensorAddBuffer(self.builder, buffer_id)
        tf_tensor = tflite.Tensor.TensorEnd(self.builder)

        tensor_id = self.EmitTensorId()
        self.tensors.append(tf_tensor)

        return tensor_id

    def SerializeOpFullyConnected(self, fused_activation_function, input_id,
                                  output_id, W_id, b_id):

        logger.info(
            "fully_connected. input = {}, output = {}, W = {}, b = {}".format(
                input_id, output_id, W_id, b_id))
        opcode_id = self.RegisterBuiltinOpcode(
            tflite.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED)

        # Options
        tflite.FullyConnectedOptions.FullyConnectedOptionsStart(self.builder)
        tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(
            self.builder, fused_activation_function)
        tf_options = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(
            self.builder)

        # Inputs
        num_inputs = 3
        tflite.Operator.OperatorStartInputsVector(self.builder, num_inputs)
        self.builder.PrependInt32(b_id)
        self.builder.PrependInt32(W_id)
        self.builder.PrependInt32(input_id)
        tf_inputs = self.builder.EndVector(num_inputs)

        # Outputs
        num_outputs = 1
        tflite.Operator.OperatorStartOutputsVector(self.builder, num_outputs)
        self.builder.PrependInt32(output_id)
        tf_outputs = self.builder.EndVector(num_outputs)

        tflite.Operator.OperatorStart(self.builder)
        tflite.Operator.OperatorAddInputs(self.builder, tf_inputs)
        tflite.Operator.OperatorAddOutputs(self.builder, tf_outputs)
        tflite.Operator.OperatorAddBuiltinOptionsType(
            self.builder,
            tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions)
        tflite.Operator.OperatorAddBuiltinOptions(self.builder, tf_options)
        logger.debug('opcode_id = {}'.format(opcode_id))
        tflite.Operator.OperatorAddOpcodeIndex(self.builder, opcode_id)
        op = tflite.Operator.OperatorEnd(self.builder)

        self.operators.append(op)

        return op

    def SerializeOpReLU(self, input_id, output_id):

        opcode_id = self.RegisterBuiltinOpcode(
            tflite.BuiltinOperator.BuiltinOperator.RELU)

        # Inputs
        num_inputs = 1
        tflite.Operator.OperatorStartInputsVector(self.builder, num_inputs)
        self.builder.PrependInt32(input_id)
        inputs = self.builder.EndVector(num_inputs)

        # Outputs
        num_outputs = 1
        tflite.Operator.OperatorStartOutputsVector(self.builder, num_outputs)
        self.builder.PrependInt32(output_id)
        outputs = self.builder.EndVector(num_outputs)

        tflite.Operator.OperatorStart(self.builder)
        tflite.Operator.OperatorAddInputs(self.builder, inputs)
        tflite.Operator.OperatorAddOutputs(self.builder, outputs)
        tflite.Operator.OperatorAddOpcodeIndex(self.builder, opcode_id)
        op = tflite.Operator.OperatorEnd(self.builder)

        self.operators.append(op)

        return op

    def SerializeOpReshape(self, input_id, output_id, new_shape):
        """Serialize Reshape function.

        Args:

            new_shape ([int]): New shape.

        """

        opcode_id = self.RegisterBuiltinOpcode(
            tflite.BuiltinOperator.BuiltinOperator.RESHAPE)

        # Options
        tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
            self.builder, len(new_shape))
        for i in reversed(new_shape):
            self.builder.PrependInt32(i)
        tf_new_shape = self.builder.EndVector(len(new_shape))

        tflite.ReshapeOptions.ReshapeOptionsStart(self.builder)
        tflite.ReshapeOptions.ReshapeOptionsAddNewShape(
            self.builder, tf_new_shape)
        tf_options = tflite.ReshapeOptions.ReshapeOptionsEnd(self.builder)

        # Inputs
        num_inputs = 1
        tflite.Operator.OperatorStartInputsVector(self.builder, num_inputs)
        self.builder.PrependInt32(input_id)
        tf_inputs = self.builder.EndVector(num_inputs)

        # Outputs
        num_outputs = 1
        tflite.Operator.OperatorStartOutputsVector(self.builder, num_outputs)
        self.builder.PrependInt32(output_id)
        tf_outputs = self.builder.EndVector(num_outputs)

        tflite.Operator.OperatorStart(self.builder)
        tflite.Operator.OperatorAddInputs(self.builder, tf_inputs)
        tflite.Operator.OperatorAddOutputs(self.builder, tf_outputs)
        tflite.Operator.OperatorAddBuiltinOptionsType(
            self.builder, tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions)
        tflite.Operator.OperatorAddBuiltinOptions(self.builder, tf_options)
        logger.debug('opcode = {}'.format(opcode_id))
        tflite.Operator.OperatorAddOpcodeIndex(self.builder, opcode_id)
        op = tflite.Operator.OperatorEnd(self.builder)

        self.connection_ids.append(tf_outputs)
        self.operators.append(op)

        return op

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
            if I.name == 'input0':
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.shape, None)
                self.inputs['input0'] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.shape, I.data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # W
            W_id = tf_serializer.SerializeTensor(parent_layer_names[1],
                                                 W.shape, W.data)

            # b
            b_id = -1  # -1 = optional
            if b is not None:
                if b.data is not None:
                    b_id = tf_serializer.SerializeTensor(
                        parent_layer_names[2], b.shape, b.data)

            # output
            _output = func.outputs[0]
            logger.info("output.shape = {}".format(_output().shape))
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            activation_function = 0  # 'NONE'
            tf_serializer.SerializeOpFullyConnected(activation_function,
                                                    input_id, output_id, W_id,
                                                    b_id)

        elif func.label == 'ReLU':

            assert (len(func.inputs) == 1)

            # input
            I = func.inputs[0]
            if I.name == 'input0':
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.shape, None)
                self.inputs['input0'] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.shape, I.data)
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
                                                      _output().shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)

            tf_serializer.SerializeOpReLU(input_id, output_id)

        elif func.label == 'Reshape':

            assert (len(func.inputs) == 1)

            new_shape = func.outputs[0]().shape

            logger.info('Reshape : {}'.format(new_shape))

            I = func.inputs[0]
            if I.name == 'input0':
                # Placeholder input
                input_id = tf_serializer.SerializeTensor(
                    I.name, I.shape, None)
                self.inputs['input0'] = input_id
            elif parent_layer_names[0] == 'data':
                input_id = tf_serializer.SerializeTensor(
                    layer_name + '_input0', I.shape, I.data)
            else:
                input_id = tf_serializer.FindConnection(parent_layer_names[0])
                # There should have valid connection
                if input_id is None:
                    logger.fatal('{} not found in connections'.format(
                        parent_layer_names[0]))
                    raise

            # output
            output_id = tf_serializer.SerializeTensor(layer_name + '_0',
                                                      new_shape, None)
            tf_serializer.RegisterConnection(layer_name, output_id)
            output_id = tf_serializer.SerializeOpReshape(
                input_id, output_id, new_shape)

        else:
            logger.error("Unknown or unsupported function/link : %s",
                         func.label)

    def __call__(self, outputs):

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
    inputs = []
    if isinstance(args, tuple):
        args = list(args)

    if isinstance(args, list):
        for i, arg in enumerate(args):
            if isinstance(arg, chainer.get_array_types()):
                input_name = 'input{}'.format(i)
                args[i] = chainer.Variable(arg, name=input_name)
                inputs.append(args[i])

        outputs = model(*args)
    elif isinstance(args, chainer.get_array_types()):
        args = chainer.Variable(args, name='input0')
        inputs.append(args)
        outputs = model(args)

    elif isinstance(args, chainer.Variable):
        # Rewrite name
        args.name = 'input0'
        inputs.append(args)
        outputs = model(args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list, tuple, '
            'numpy array, or Chainer Variable. But a {} object was '
            'given.'.format(type(args)))


    if isinstance(outputs, variable.Variable):
        outputs = [outputs]
    assert isinstance(outputs, (tuple, list))
    for i in outputs:
        assert isinstance(i, variable.Variable)

    converter = TensorFlowLiteConverter(filename)
    converter(outputs)

    # Chainer's result
    return inputs, outputs
