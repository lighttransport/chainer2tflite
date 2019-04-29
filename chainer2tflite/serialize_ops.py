# -*- coding: utf-8 -*-

from . import tflite

from .tflite import FullyConnectedOptions
from .tflite import Operator
from .tflite import OperatorCode
from .tflite import BuiltinOperator
from .tflite import BuiltinOptions
from .tflite import AddOptions
from .tflite import ReshapeOptions
from .tflite import ResizeBilinearOptions
from .tflite import Pool2DOptions
from .tflite import PadOptions

from .tflite import ActivationFunctionType
from .tflite import Padding


def SerializeOpFullyConnected(serializer, fused_activation_function, input_id,
                              output_id, W_id, b_id):

    serializer.logger.info(
        "fully_connected. input = {}, output = {}, W = {}, b = {}".format(
            input_id, output_id, W_id, b_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    tflite.FullyConnectedOptions.FullyConnectedOptionsStart(serializer.builder)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tf_options = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 3
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(b_id)
    serializer.builder.PrependInt32(W_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op

def SerializeAveragePooling2D(serializer, input_id,
                              output_id, fused_activation_function, padding, stride, filter_size):
    """Serialize average_pooling_2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        filter_size([int]): [filter_width, filter_height].

    """

    serializer.logger.info(
        "average_pooling_2d. input = {}, output = {}, fused_activation_function = {}, stride = {}, filter_size = {}".format(
            input_id, output_id, fused_activation_function, stride, filter_size))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.AVERAGE_POOL_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise


    tflite.Pool2DOptions.Pool2DOptionsStart(serializer.builder)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(
        serializer.builder, padding_type)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(
        serializer.builder, stride[0])
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(
        serializer.builder, stride[1])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(
        serializer.builder, filter_size[0])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(
        serializer.builder, filter_size[1])
    tf_options = tflite.Pool2DOptions.Pool2DOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op

def SerializeMaxPooling2D(serializer, input_id,
                          output_id, fused_activation_function, padding, stride, filter_size):
    """Serialize max_pooling_2d.

    Args:
        serializer: tflite serializer.
        input_id(int): Input Tensor id.
        output_id(int): Output Tensor id.
        fused_activation_function(string): activation function type('NONE').
        padding(string): padding('SAME' or 'VALID')
        stride([int]): [stride_w, stride_h].
        filter_size([int]): [filter_width, filter_height].

    """

    serializer.logger.info(
        "max_pooling_2d. input = {}, output = {}, fused_activation_function = {}, stride = {}, filter_size = {}".format(
            input_id, output_id, fused_activation_function, stride, filter_size))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.MAX_POOL_2D)

    # Options
    if fused_activation_function == 'NONE':
        activation_function_type = tflite.ActivationFunctionType.ActivationFunctionType.NONE
    else:
        print('Unsupported activation function: ', fused_activation_function)
        raise

    if padding == 'VALID':
        padding_type = tflite.Padding.Padding.VALID
    elif padding == 'SAME':
        padding_type = tflite.Padding.Padding.SAME
    else:
        print('Unsupported padding: ', padding)
        raise


    tflite.Pool2DOptions.Pool2DOptionsStart(serializer.builder)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(
        serializer.builder, activation_function_type)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(
        serializer.builder, padding_type)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(
        serializer.builder, stride[0])
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(
        serializer.builder, stride[1])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(
        serializer.builder, filter_size[0])
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(
        serializer.builder, filter_size[1])
    tf_options = tflite.Pool2DOptions.Pool2DOptionsEnd(
        serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.Pool2DOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpResizeImages(serializer, input_id, output_id, new_shape_id):

    # NOTE(LTE): Chainer supports bilinear interpolation only.
    # Map to resize_bilinear + align_corners = true.
    # For more details about resize_images,
    # See https://github.com/chainer/onnx-chainer/issues/147

    serializer.logger.info(
        "resize_images. input = {}, output = {}, new_shape = {}".format(
            input_id, output_id, new_shape_id))
    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESIZE_BILINEAR)

    # Options
    # (`align_corners` == true) matches the Chainer's result.
    tflite.ResizeBilinearOptions.ResizeBilinearOptionsStart(serializer.builder)
    tflite.ResizeBilinearOptions.ResizeBilinearOptionsAddAlignCorners(
        serializer.builder, True)
    tf_options = tflite.ResizeBilinearOptions.ResizeBilinearOptionsEnd(
        serializer.builder)

    # Inputs
    # new_shape first.
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(new_shape_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ResizeBilinearOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode_id = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpAdd(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.ADD)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    # Options
    activation_function = 0  # 'NONE'
    tflite.AddOptions.AddOptionsStart(serializer.builder)
    tflite.AddOptions.AddOptionsAddFusedActivationFunction(
        serializer.builder, activation_function)
    tf_options = tflite.AddOptions.AddOptionsEnd(serializer.builder)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.AddOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpSub(serializer, x_id, y_id, output_id):

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.SUB)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(y_id)
    serializer.builder.PrependInt32(x_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op

def SerializeOpPad(serializer, input_id, output_id, padding_id):
    """Serialize Pad.

    Args:

        input_id (int): Input tensor id.
        output_id (int): Output tensor id.
        padding_id (int): Tensor id which contains padding size(2x2 shape).

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.PADV2)

    # Options
    # Currently PadOptions has empty parameter.
    tflite.PadOptions.PadOptionsStart(serializer.builder)
    tf_options = tflite.PadOptions.PadOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 2
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(padding_id)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.PadOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpReshape(serializer, input_id, output_id, new_shape):
    """Serialize Reshape function.

    Args:

        new_shape ([int]): New shape.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESHAPE)

    # Options
    tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
        serializer.builder, len(new_shape))
    for i in reversed(new_shape):
        serializer.builder.PrependInt32(i)
    tf_new_shape = serializer.builder.EndVector(len(new_shape))

    tflite.ReshapeOptions.ReshapeOptionsStart(serializer.builder)
    tflite.ReshapeOptions.ReshapeOptionsAddNewShape(serializer.builder,
                                                    tf_new_shape)
    tf_options = tflite.ReshapeOptions.ReshapeOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder,
        tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op


def SerializeOpReLU(serializer, input_id, output_id):
    """Serialize ReLU op.

    Args:
        serializer(TensorFlowLiteSerializer):
        input_id(int32): Input tensor id
        output_id(int32): Output tensor id

    Returns:
        tflite.Operator

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RELU)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, outputs)
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op

def SerializeOpReshape(serializer, input_id, output_id, new_shape):
    """Serialize Reshape function.

    Args:

        input_id (int): Input Tensor id.
        output_id (int): Output Tensor id.
        new_shape ([int]): New shape.

    """

    opcode_id = serializer.RegisterBuiltinOpcode(
        tflite.BuiltinOperator.BuiltinOperator.RESHAPE)

    # Options
    tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(
        serializer.builder, len(new_shape))
    for i in reversed(new_shape):
        serializer.builder.PrependInt32(i)
    tf_new_shape = serializer.builder.EndVector(len(new_shape))

    tflite.ReshapeOptions.ReshapeOptionsStart(serializer.builder)
    tflite.ReshapeOptions.ReshapeOptionsAddNewShape(
        serializer.builder, tf_new_shape)
    tf_options = tflite.ReshapeOptions.ReshapeOptionsEnd(serializer.builder)

    # Inputs
    num_inputs = 1
    tflite.Operator.OperatorStartInputsVector(serializer.builder, num_inputs)
    serializer.builder.PrependInt32(input_id)
    tf_inputs = serializer.builder.EndVector(num_inputs)

    # Outputs
    num_outputs = 1
    tflite.Operator.OperatorStartOutputsVector(serializer.builder, num_outputs)
    serializer.builder.PrependInt32(output_id)
    tf_outputs = serializer.builder.EndVector(num_outputs)

    tflite.Operator.OperatorStart(serializer.builder)
    tflite.Operator.OperatorAddInputs(serializer.builder, tf_inputs)
    tflite.Operator.OperatorAddOutputs(serializer.builder, tf_outputs)
    tflite.Operator.OperatorAddBuiltinOptionsType(
        serializer.builder, tflite.BuiltinOptions.BuiltinOptions.ReshapeOptions)
    tflite.Operator.OperatorAddBuiltinOptions(serializer.builder, tf_options)
    serializer.logger.debug('opcode = {}'.format(opcode_id))
    tflite.Operator.OperatorAddOpcodeIndex(serializer.builder, opcode_id)
    op = tflite.Operator.OperatorEnd(serializer.builder)

    serializer.operators.append(op)

    return op
