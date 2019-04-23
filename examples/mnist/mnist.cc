#include <cstdlib>
#include <cstdio>
#include <iostream>

#ifdef __clang__
#pragma clang diagnostic push "-Weverything"
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "mnist (mnist.tflite) (path_to_minist_data)" << std::endl;
    return EXIT_FAILURE;
  }

  std::string input_modelfilename = "../mnist.tflite";
  if (argc > 1) {
    input_modelfilename = argv[1];
  }

  std::string mnist_data_path = "./data";
  if (argc > 2) {
    mnist_data_path = argv[2];
  }

  std::cout << "Input model filename : " << input_modelfilename << "\n";
  std::cout << "MNIST Data path      : " << mnist_data_path << "\n";

  std::unique_ptr<tflite::FlatBufferModel> model;
  model = tflite::FlatBufferModel::BuildFromFile(input_modelfilename.c_str());
  if (!model) {
    std::cerr << "failed to load model. filename = " << input_modelfilename << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Loaded model : " << input_modelfilename << "\n";
  model->error_reporter();

  tflite::ops::builtin::BuiltinOpResolver resolver;

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    std::cerr << "failed to construct interpreter.\n";
    return EXIT_FAILURE;
  }

  // interpreter->UseNNAPI(1);

  return EXIT_SUCCESS;
}
