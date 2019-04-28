import glob
import os
import warnings

import numpy as np

try:
    import tensorflow as rt
    TFLITE_AVAILABLE = True
except ImportError:
    warnings.warn(
        'tensorflow not found. Please install TensorFlow to use '
        ' the testing utility for Chainer2TFlite\'s converters.',
        ImportWarning)
    TFLITE_AVAILABLE = False


def load_test_data(data_dir, input_names, output_names):
    inout_values = []
    for kind, names in [('input', input_names), ('output', output_names)]:
        names = list(names)
        values = {}
        print('kind = ', kind, ', names = ', names)

        files = glob.glob(os.path.join(data_dir, '{}_*.npy'.format(kind)))

        assert len(files) == len(names)

        for npy in sorted(files):
            print(npy)
            arr = np.load(npy)
            name = names.pop(0)
            values[name] = arr
        inout_values.append(values)
    return tuple(inout_values)


def check_model_expect(test_path, input_names=None):
    if not TFLITE_AVAILABLE:
        raise ImportError('TensorFlow is not found on checking module.')

    model_path = os.path.join(test_path, 'model.tflite')

    # Load TFLite model and allocate tensors.
    interpreter = rt.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    _output_details = interpreter.get_output_details()

    # NOTE(LTE): output_details may contain duplicated items due to tensorflow bug?.
    # As a work around, create unique entries of outputs

    output_details = []
    for outp in _output_details:
        # Simple linear search
        found = False
        for i in range(len(output_details)):
            if outp['name'] == output_details[i]['name']:
                found = True


        if not found:
            output_details.append(outp)

    print('input_details', input_details)
    print('output_details', output_details)

    # TODO(LTE): Support multiple output tensors
    assert len(output_details) == 1

    rt_input_names = [value['name'] for value in input_details]
    rt_output_names = [value['name'] for value in output_details]

    print('rt inputs', rt_input_names)
    print('rt outputs', rt_output_names)

    # To detect unexpected inputs created by exporter, check input names
    if input_names is not None:
        assert list(sorted(input_names)) == list(sorted(rt_input_names))

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])

    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)

        print('dbg: ', test_data_path)
        print('input_names: ', rt_input_names)
        print('output_names: ', rt_output_names)

        inputs, outputs = load_test_data(
            test_data_path, rt_input_names, rt_output_names)

        print('ref in', inputs)
        print('ref out', outputs)

        # TODO(LTE): Support multiple output tensors
        assert len(outputs) == 1

        # Set input tensors
        for ref_name, ref_value in inputs.items():

            if len(ref_value.shape) == 4:
                # Assume (batch, C, H, W)
                # Convert to (batch, H, W, C)(TensorFlow's default)
                ref_value = np.transpose(ref_value, (0, 2, 3, 1))
                print('shape = ', ref_value.shape)

            # Simple linear search
            found = False
            for i in range(len(input_details)):
                if ref_name == input_details[i]['name']:
                    interpreter.set_tensor(input_details[i]['index'], ref_value)
                    found = True

            if not found:
                print('input[{}] not found in tflite inputs'.format(ref_name))
                raise


        interpreter.invoke()

        rt_out = interpreter.get_tensor(output_details[0]['index'])

        print('tflite out', rt_out)

        cn_out = outputs[rt_output_names[0]]

        if len(rt_out.shape) == 4:
            # Convert to NCHW for comparison with Chainer's result
            rt_out = np.transpose(rt_out, (0, 3, 1, 2))

        # Its a dinner time! Taste it!
        for cy, my in zip(cn_out, rt_out):
            np.testing.assert_allclose(cy, my, rtol=1e-5, atol=1e-5)
