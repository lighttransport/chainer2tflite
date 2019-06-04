# Chainer to TensorFlow-Lite converter

`chainer2tflite` is a model converter from Chainer(Python + NPZ) to TensorFlow-Lite(flatbuffers format)

## Status

In-progress. At least MNIST with 2 LinearFunction nodes works.

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

## NCHW and NHWC conversion

TensorFlow-Lite only supports NHWC format.
Chainer uses NCHW as a default tensor format.

chainer2tflite inserts `Transpose` op for each input and output of op where NHWC format conversion is required(e.g. `CONV_2D`).
This is redundant but simple and workable solution.
Insert minimum `Transpose` op by analyzing input graph is TODO.


## Supported layers/ops

| Chainer                    | tflite                       | Comment                                     |
| -------------------------- | ---------------------------- | ------------------------------------------- |
| Add                        | ADD                          | two inputs only                             |
| Sub                        | SUB                          |                                             |
| Mul                        | MUL                          |                                             |
| Div                        | DIV                          |                                             |
| Reshape                    | RESHAPE                      |                                             |
| LinearFunction             | FULLY_CONNECTED              | activation=None                             |
| ReLU                       | RELU                         |                                             |
| LeakyReLU                  | REAKY_RELU                   |                                             |
| ResizeImages               | RESIZE_BILINEAR              | `align_corners=true`                        |
| Pad                        | PADV2                        | Support constant value                      |
| Sigmoid                    | LOGISTIC                     |                                             |
| AveragePooling2D           | AVERAGE_POOL_2D              |                                             |
| MaxPooling2D               | MAX_POOL_2D                  |                                             |
| Convolution2D              | CONV_2D                      | dilated=1                                   |
| DilatedConvolution2D       | CONV_2D                      | dilated=N                                   |
| SoftMax                    | SOFTMAX                      | `axis` in Chainer must be last dim          |
| LogSoftMax                 | LOG_SOFTMAX                  | `axis` in Chainer must be last dim          |
| Deconvolution2D            | CONV_2D_TRANSPOSE            |                                             |
| Vstak                      | CONCATENATION OR PACK(1D)    | axis=0                                      |
| Hstak                      | CONCATENATION                | axis=1                                      |
| Unpooling2D                | RESIZE_NEAREST_NEIGHBOR      | integer scaling factor only(e.g. 2.0, 3.0)  |
| Transpose                  | TRANSPOSE                    |                                             |
| Squeeze                    | SQUEEZE                      |                                             |
| Tile                       | TILE                         |                                             |
| ExpandDims                 | EXPAND_DIMS                  |                                             |
| Concat                     | CONCATENATION                |                                             |
| LocalResponseNormalization | LOCAL_RESPONSE_NORMALIZATION |                                             |
| DepthwiseConvolution2D     | DEPTHWISE_CONV2D             | filter(weight) should be constant           |


## Conditionally supported layers/ops

| Chainer                 | tflite                    | Comment                                     |
| ----------------------- | ------------------------- | ------------------------------------------- |
| Dropout                 | See comments              | Use deterministic random-valued tensor      |
| ELU                     | ELU                       | tflite `r1.14` or later                     |
| Space2Depth             | SPACE_TO_DEPTH            | 4D Tensor only                              |
| Cast                    | CAST                      | `float32`, `int32` only                     |
| NormalizeL2(normalize)  | L2_NORMALIZATION          | axis must be the last dim of input tensor.  |
| DepthwiseConvolution2D  | DEPTHWISE_CONV_2D         | See details described the below             |

See `chainer2tflite/convert_dropout.py` for details on `Dropout` conversion.
At least we've confirmed unit test passes by force setting same seed value for random number generation.

For `DepthwiseConvolution2D`, Chainer lowers it to `Transpose`, `Reshape` and `Convolution2DFunction` function sequences.
We try to reconstruct `DepthwiseConvolution2D` from such function sequences, but may not be perfect.

## Unsupported Chainer functions

### floor, ceil

It looks Chainer directly calls `floor`, `ceil` math function and we cannot retrieve symbolic reposentation of these node.
Although tflite serializer(`chainer2tflite/serialize_ops.py`) supports `FLOOR` and `CEIL`

### Untested layers/ops

* Conv2D with stride > 1
* Ave/Max pooling with stride > 1

### Not supported(or TODO)

* Pooling
  * [ ] UnpoolingND
  * [ ] ConvND
  * [ ] PooingND
  * [ ] ROIPooing2D
* [ ] Absolute and other primitive math expression.
  * [ ] sqrt
  * [ ] mean
  * [ ] mean_squared_error
* [ ] ADD_N
* [ ] ARG_MAX, ARG_MIN
* Normalization
  * [ ] BatchNormalization
    * Decompose into some math ops.
  * [ ] FixedBatchNormalization
* Array
  * [ ] Depth2Space(tflite does not support yet)
  * [ ] SplitAxis
  * [ ] Where
  * [ ] Copy
  * [ ] GetItem
* RNN
  * [ ] RNN
  * [ ] GRU
  * [ ] LSTM
* Loss
  * [ ] SoftmaxCrossEntropy
* [ ] TOP_K
* [ ] T.B.W.

## For developers

### Inspect generated model

You can use Netron: https://github.com/lutzroeder/netron/

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

* Unit tests
  * [ ] Reshape
* [ ] RNN
* [ ] Remove redundant NCHW and NHWC conversion.
  * For example, Currently we insert `Transpose` op for each `Conv2D` op. When the network contains sequence of `Conv2D`, we should insert `Transpose` for the beggining and the end of `Conv2D`s.
* [ ] Support a node which outputs multiple tensors(e.g. `split`)
* [ ] Support a graph which have multiple outputs.
* [ ] Android demo(train with Chainer, run tflite on mobile).
* [ ] Support TensorFlow-Lite micro(experimental) to run Chainer-trained model on IoT devices
* [ ] More functions/links, ops, etc
* [ ] Quantize weight
* [ ] Refactor unit tester
* [ ] Write tflite model to memory
* [ ] Test if its safe we can convert Conv2D with `group > 1` to DEPTHWISE_CONV2D

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
