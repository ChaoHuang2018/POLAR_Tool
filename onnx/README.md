## Parsing an ONNX file

### Requirement

* Install python onnx library `pip install onnx`

### Usage
* Copy the `onnx_converger` file to the same directory of the C++ source file.
* In C++ source file, use command `system("python onnx_converter $ONNX_FILE_PATH");` to generatge POLAR input neural network file. The generated file has the same name as the onnx file except for the `.onnx` suffix.