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

| Chainer         | tflite           | Comment              |
| --------------- | ---------------- | -------------------- |
| Reshape         | RESHAPE          |                      |
| LinearFunction  | MATMUL           |                      |
| ReLU            | RELU             |                      |
| Softmax         | SOFTMAX          |                      |
| Conv2D          | CONV_2D          |                      |
| ResizeImages    | RESIZE_BILINEAR  | `align_corners=true` |

## Tests

### Requirements

* pytest

### Running tests

```
$ pytest tests/
```

## Examples

See `examples` directory.

## TODO

* [ ] Android demo
* [ ] Support TensorFlow-Lite micro(experimental) to run Chainer-trained model on IoT devices

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
