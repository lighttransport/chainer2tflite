# helper.py is based on onnx-chainer/tests/helper.py

# MIT License
#
# Copyright (c) 2017 Shunta Saito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import unittest
import warnings

import tensorflow.lite.python
import pytest

from chainer2tflite.testing.get_test_data_set import gen_test_data_set


class TFLiteModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_config(self, disable_experimental_warning):
        pass

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()
        self.check_out_values = None
        selected_runtime = request.config.getoption('value-check-runtime')

        from chainer2tflite.testing.test_tflite import check_model_expect  # NOQA
        self.check_out_values = check_model_expect


    def expect(self, model, args, name=None,
               with_warning=False, input_names=None, output_names=None):

        test_name = name
        if test_name is None:
            test_name = self.default_name

        dir_name = 'test_' + test_name
        if with_warning:
            with warnings.catch_warnings(record=True) as w:
                test_path = gen_test_data_set(
                    model, args, dir_name,
                    input_names, output_names)
            assert len(w) == 1
        else:
            test_path = gen_test_data_set(
                model, args, dir_name, opset_version, train, input_names,
                output_names)


        # Export function can be add unexpected inputs. Collect inputs
        # from tflite model, and compare with another input list got from
        # test runtime.
        if self.check_out_values is not None:
            self.check_out_values(test_path, input_names=graph_input_names)
