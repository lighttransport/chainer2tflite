# Chainer to TensorFlow-Lite converter

`chainer2tflite` is a model converter from Chainer(Python + NPZ) to TensorFlow-Lite(flatbuffers format)

## Requirements

* Chainer v5.4 or later
* TensorFlow `r1.13.1` or later `r1.x` branch
* flatbuffers 1.10 or later
* Python 3.6 or later

## Setup

```
$ pip install chainer
$ pip install flatbuffers
```

### For developers only

Python binding of tflite schema(version 3) is already added to `tflite` directory.
If you want to re-generate python binding from tflite schema, use `flatc` to generate it:

```
$ flatc -p /path/to/tensorflow/tensorflow/lite/schema/schema.fbs
```


## Supported layers/ops

| Chainer         | tflite           | Comment                 |
| --------------- | ---------------- | ----------------------- |
| Add             | ADD              | two inputs only         |
| Reshape         | RESHAPE          |                         |
| LinearFunction  | FULLY_CONNECTED  | activation=None         |
| ReLU            | RELU             |                         |
| ResizeImages    | RESIZE_BILINEAR  | `align_corners=true`    |

### Not supported(or TODO)

* [ ] Conv2D
* [ ] Deconv2D
* [ ] FusedBatchNormalization
* [ ] Absolute and other primitive math expression.
* [ ] ADD_N
* [ ] ARG_MAX, ARG_MIN
* [ ] T.B.W.

## Tests

### Requirements

* pytest
* tensorflow(CPU version preferred)

```
$ pip install pytest
$ pip install tensorflow
```

### Running tests

```
$ pytest tests/
```

#### Note on test data

Input/output test data will be generated in `out` directory.
Remove this directory if you added/modified unit tests.

### For developers

#### Linting python code

```
$ pip install hacking flake8-import-order flake8-docstrings
```

```
$ flake8
```

#### Formatting python code

```
$ pip install yapf
```

```
$ yapf <input.py>
```

## Examples

See `examples` directory.

## TODO

* [ ] Automatic handling of NCHW and NHWC conversion.
* [ ] Support multiple outputs graph.
* [ ] Android demo(train with Chainer, run tflite on mobile).
* [ ] Support TensorFlow-Lite micro(experimental) to run Chainer-trained model on IoT devices
* [ ] More functions/links, ops, etc
* [ ] Quantized network
* [ ] Refactor unit tester
* [ ] Write tflite model to memory

## License

MIT license.

### Thrid party licenses.

`chainer2tflite` uses some python codes from Chainer and `onnx-chainer`. Chainer and `onnx-chainer` is both licensed under MIT license.


#### onnx-chainer

```
MIT License

Copyright (c) 2017 Shunta Saito

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
