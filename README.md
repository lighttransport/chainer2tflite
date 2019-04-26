# Chainer to TensorFlow-Lite converter

`chainer2tflite` is a model converter from Chainer(Python + NPZ) to TensorFlow-Lite(flatbuffers format)

## Requirements

* Chainer v5.4 or later
* TensorFlow `r1.13.1` or later
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

## Examples

See `examples` directory.

## TODO

* [ ] Android demo
* [ ] Support TensorFlow-Lite micro(experimental) to run Chainer-trained model on IoT devices

## License

MIT license.

### Thrid party licenses.

* `chainer2tflite` uses some python codes from Chainer. Chainer is licensed under MIT license.
