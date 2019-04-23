# Chainer to TensorFlow-Lite converter

`chainer2tflite` is a model converter from Chainer to TensorFlow-Lite

## Requirements

* Chainer v5.4 or later
* TensorFlow `r1.13` or later
* flatbuffers 1.10 or later
* Python 3.6 or later

## Setup

```
$ pip install chainer
$ pip install flatbuffers
```

### For developers only

Generate python binding from tflite schema.

```
$ sh gen_tflite_from_schema.sh
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

## TODO

* [ ] Android demo
* [ ] Support TensorFlow-Lite micro(experimental) to run Chainer model on IoT devices

## License

MIT license.

### Thrid party licenses.

* `chainer2tflite` uses some python codes from Chainer. Chainer is licensed under MIT license.
