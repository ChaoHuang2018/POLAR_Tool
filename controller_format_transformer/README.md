<!--
### Instructions: Transform an original txt-format neural network to onnx one

python txt2onnx.py -i <inputfile> -o <outputfile>

inputfile: the file name of the input txt neural network

outputfile: the file name of the output onnx neural network
-->


### Instructions: Transform an original txt-format neural network to Pytorch model and then to onnx one
* Make sure that onnx and onnxruntime are install, i.e. pip install xxxx
 
* Run 
```
python torch2onnx.py <inputfile>
```

* The outputfile has the same file name as inputfile except for the `.onnx` suffix.


### Instructions: Transform an original onnx neural network to txt-format model

* Run 
```
python keras2txt.py --keras_model_file=<inputfile> --output_txt_model_file=<outputfile>
```
