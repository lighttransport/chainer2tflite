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
# in `chainer2tflite` directory,
$ flatc -p /path/to/tensorflow/tensorflow/lite/schema/schema.fbs
```


## Supported layers/ops

| Chainer                 | tflite            | Comment                                  |
| ----------------------- | ----------------- | ---------------------------------------- |
| Add                     | ADD               | two inputs only                          |
| Reshape                 | RESHAPE           |                                          |
| LinearFunction          | FULLY_CONNECTED   | activation=None                          |
| ELU                     | ELU               | tflite `r1.14` or later                  |
| ReLU                    | RELU              |                                          |
| LeakyReLU               | REAKY_RELU        |                                          |
| ResizeImages            | RESIZE_BILINEAR   | `align_corners=true`                     |
| Pad                     | PADV2             | Support constant value                   |
| AveragePooling2D        | AVERAGE_POOL_2D   |                                          |
| MaxPooling2D            | MAX_POOL_2D       |                                          |
| Convolution2D           | CONV_2D           | dilated=1                                |
| DilatedConvolution2D    | CONV_2D           | dilated=N                                |
| SoftMax                 | SOFTMAX           | axis in Chainer must be last dim         |
| LogSoftMax              | LOG_SOFTMAX       | axis in Chainer must be last dim         |
| Deconvolution2D         | CONV_2D_TRANSPOSE |                                          |

### Untested layers/ops

* Conv2D with stride > 1
* Ave/Max pooling with stride > 1

### Not supported(or TODO)

* [ ] Unpooling2D
* [ ] ConvND
* [ ] PooingND
* [ ] hstack, vstack(use `Pack`?)
* Random
  * [ ] Dropout
* [ ] Absolute and other primitive math expression.
  * [ ] sqrt
  * [ ] mean
  * [ ] mean_squared_error
* [ ] ADD_N
* [ ] ARG_MAX, ARG_MIN
* Normalization
  * [ ] BatchNormalization
  * [ ] FixedBatchNormalization
  * [ ] LocalResponseNormalization
  * [ ] NormalizeL2
* Array
  * [ ] Depth2Space, Space2Depth
  * [ ] SplitAxis
  * [ ] Squeeze
  * [ ] Tile
  * [ ] Transpose
  * [ ] ExpandDims
  * [ ] Where
  * [ ] Cast
  * [ ] Concat
  * [ ] Copy
  * [ ] GetItem
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
