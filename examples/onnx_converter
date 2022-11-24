#!/usr/bin/python

import onnx
from onnx import numpy_helper
import os, sys
 
 


def main(onnx_model_path):
    # Or you can load a regular onnx model and pass it to the converter
    onnx_model = onnx.load(onnx_model_path)
    #torch_model = convert(onnx_model) 
     
    poloar_model_path = onnx_model_path.split(".onnx")[0]
    
    
    wbs = {}
    layers = [(initializer.name, numpy_helper.to_array(initializer)) for initializer in onnx_model.graph.initializer]
    for ((wname, w), (bname, b)) in zip(layers[::2], layers[1::2]):
        #print(wname, bname)
        wbs[(wname, bname)] = (w, b)

    wb_ks = list(wbs.keys())
    wb_vals = [] 
    activs = []
    
    for node in onnx_model.graph.node:
        for (wname, bname) in wb_ks:
            if wname in node.input and bname in node.input:
                wb_vals.append(wbs[(wname, bname)])
                wb_ks.remove((wname, bname))
                #print((wname, bname))
                 
                #print(node.output)
                for output in node.output:
                    if 'Relu' in output:
                        activs.append("ReLU")
                    elif 'Tanh' in output:
                        activs.append('tanh')
                    elif 'Sigmoid' in output:
                        activs.append('sigmoid')
                    else:
                        activs.append('Affine')
                    #print(activs[-1])
    
 
     
    f = open(poloar_model_path, 'w')
    num_inputs = wb_vals[0][0].shape[1]
    f.write("{}".format(num_inputs) + os.linesep)
    num_outputs = wb_vals[-1][0].shape[0]
    f.write(str(num_outputs) + os.linesep)

    num_of_hidden_layers = len(activs) - 1
    
    f.write(str(num_of_hidden_layers) + os.linesep)

    for i in range(num_of_hidden_layers):
        f.write(str(wb_vals[i][1].shape[0]) + os.linesep) 

    
    for activ in activs:
        f.write(str(activ) + os.linesep)

    count = 0
    for (w, b) in wb_vals:
        #print(w.shape, b.shape)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                f.write(str(w[i][j]) + os.linesep)
                count += 1
            f.write(str(b[i]) + os.linesep)
            count += 1
    #print(count)            
 
    f.write("0.0" + os.linesep)
    f.write("1.0")

    f.close()
 
def test():
    import torch
    class NN(torch.nn.Module):
        def __init__(self):
            # call constructor from superclass
            super().__init__()
            
            # define network layers
            self.fc1 = torch.nn.Linear(16, 12)
            self.fc2 = torch.nn.Linear(12, 10)
            self.fc3 = torch.nn.Linear(10, 1)
            
        def forward(self, x):
            # define forward pass
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.sigmoid(self.fc3(x))
            return x
    nn = NN()
    def init_normal(module):
        if type(module) == torch.nn.Linear:
            torch.nn.init.normal_(module.weight, mean=0, std=0.01)
            torch.nn.init.zeros_(module.bias)
    nn.apply(init_normal)
     
     
    import onnx

    onnx_model_path = os.path.join("test.onnx")
    torch.onnx.export(nn,               # model being run
                  torch.zeros([1, 16]),                         # model input (or a tuple for multiple inputs)
                  onnx_model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    print(f"Convert to onnx file {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    return onnx_model_path

if __name__ == "__main__":
    onnx_model_path = sys.argv[1] #test() #sys.argv[1]

    main(onnx_model_path)