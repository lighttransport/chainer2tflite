# mnist example

## Supported platform

* [x] Linux x64
* [ ] AARCH64(Android and Linux)
* [ ] Windows and macOS may work

## How to compile

Build TensorFlow Lite in some way.
We recommend forked repo https://github.com/syoyo/tensorflow with cmake build support for TensorFlow Lite(See `tensorflow/tensorflow/lite/tools/make`)

## Build

Edit path to TensorFlow-Lite and flatbuffers in `bootstrap-cmake.sh`, then

```
$ bootstrap-cmake.sh
$ cd build
$ make
$ cd ..
```

## Prepare MNIST dataset

Download and extract MINST dataset to `data` directory.

```
$ mkdir data
$ cd data
$ ../download-minist.sh
$ gunzip *.gz
```

## Run

```
$ cd build
$ ./mnist (mnist.tflite) (mnist_data_directory)
```
