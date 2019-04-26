import glob
import os
import warnings

import numpy as np

try:
    import tensorflow as rt
    TFLITE_AVAILABLE = True
except ImportError:
    warnings.warn(
        'tensorflow.lite.python not found. Please install TensorFlow to use '
        ' the testing utility for Chainer2TFlite\'s converters.',
        ImportWarning)
    TFLITE_AVAILABLE = False


def load_test_data(data_dir, input_names, output_names):
    # TODO(LTE): Implement
    return None
    #inout_values = []
    #for kind, names in [('input', input_names), ('output', output_names)]:
    #    names = list(names)
    #    values = {}
    #    for pb in sorted(
    #            glob.glob(os.path.join(data_dir, '{}_*.pb'.format(kind)))):
    #        tensor = onnx.load_tensor(pb)
    #        if tensor.name in names:
    #            name = tensor.name
    #            names.remove(name)
    #        else:
    #            name = names.pop(0)
    #        values[name] = onnx.numpy_helper.to_array(tensor)
    #    inout_values.append(values)
    #return tuple(inout_values)


def check_model_expect(test_path, input_names=None):
    if not TFLITE_AVAILABLE:
        raise ImportError('TensorFlow is not found on checking module.')

    model_path = os.path.join(test_path, 'model.tflite')

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # rt_input_names = [value.name for value in sess.get_inputs()]
    # rt_output_names = [value.name for value in sess.get_outputs()]

    # # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])


    # TODO(LTE): Implement
    raise
    """
    sess = rt.InferenceSession(onnx_model.SerializeToString())

    # To detect unexpected inputs created by exporter, check input names
    if input_names is not None:
        assert list(sorted(input_names)) == list(sorted(rt_input_names))

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])
    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)
        inputs, outputs = load_test_data(
            test_data_path, rt_input_names, rt_output_names)

        rt_out = sess.run(list(outputs.keys()), inputs)
        for cy, my in zip(outputs.values(), rt_out):
            np.testing.assert_allclose(cy, my, rtol=1e-5, atol=1e-5)
    """
