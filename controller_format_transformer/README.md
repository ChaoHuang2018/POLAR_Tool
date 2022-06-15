### Instructions: Transform an original txt-format neural network to onnx one

txt2onnx.py -i <inputfile> -o <outputfile>

inputfile: the file name of the input txt neural network

outputfile: the file name of the output onnx neural network



### Transform an original txt-format neural network to Pytorch model and then to onnx one
* Make sure that onnx and onnxruntime are install, i.e. pip install xxxx
 
* Run 'python torch2onnx.py inputfile'

* The outputfile has the same file name as inputfile except for the `.onnx` suffix.

