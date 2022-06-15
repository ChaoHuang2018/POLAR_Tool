import numpy as np
import onnx
import sys, getopt
import os
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

class Layer(object):
    def __init__(
        self,
        act,
        w,
        b
    ):
        self.act = act
        self.w = w
        self.b = b
    

class NeuralNetwork(object):
    """
    a neural network class
    """
    def __init__(
        self,
        file
    ):
        self.read_txt(file)
            
            
    def read_txt(self, input_file):
        with open(input_file) as inputfile:
            lines = inputfile.read().splitlines()

        index = 0

        self.num_of_inputs = int(lines[index])
        index = index + 1

        self.num_of_outputs = int(lines[index])
        index = index + 1

        self.num_of_hidden_layers = int(lines[index])
        index = index + 1

        network_structure = np.empty(self.num_of_hidden_layers+1, dtype=np.int32)
        for i in range(self.num_of_hidden_layers):
            network_structure[i] = int(lines[index])
            index = index + 1
        network_structure[-1] = self.num_of_outputs

        activation_list = []
        for i in range(self.num_of_hidden_layers+1):
            activation_list.append(lines[index])
            index = index + 1

        self.layers = []

        weight0 = np.empty((network_structure[0], self.num_of_inputs), dtype=np.float32)
        bias0 = np.empty((network_structure[0], 1), dtype=np.float32)
        for i in range(network_structure[0]):
            for j in range(self.num_of_inputs):
                weight0[i, j] = float(lines[index])
                index = index + 1
            bias0[i, 0] = float(lines[index])
            index = index + 1

        input_layer = Layer(activation_list[0], weight0, bias0)
        self.layers.append(input_layer)

        for layer_idx in range(self.num_of_hidden_layers):
            weight = np.empty((network_structure[layer_idx+1], network_structure[layer_idx]), dtype=np.float32)
            bias = np.empty((network_structure[layer_idx+1], 1), dtype=np.float32)

            for i in range(network_structure[layer_idx+1]):
                for j in range(network_structure[layer_idx]):
                    weight[i, j] = float(lines[index])
                    index = index + 1
                bias[i, 0] = float(lines[index])
                index = index + 1

            hidden_layer = Layer(activation_list[layer_idx+1], weight, bias)
            self.layers.append(hidden_layer)

        self.offset = float(lines[index])
        index = index + 1

        self.scale_factor = float(lines[index])

    def write_onnx(self, output_file):
        model_nodes = []
        layer_output_name_list = []

        model_input = "state"
        state = helper.make_tensor_value_info(model_input, TensorProto.FLOAT, [self.num_of_inputs])
        layer_output_name_list.append(model_input)

        model_output = "control"
        control = helper.make_tensor_value_info(model_output, TensorProto.FLOAT, [self.num_of_outputs])


        initializer_list = []
        for layer_idx in range(len(self.layers)):
            hidden_layer = self.layers[layer_idx]

            # linear transformation
            hidden_linear_output_name = "layer_linear_output" + str(layer_idx)

            weight = numpy_helper.from_array(hidden_layer.w, name="weight" + str(layer_idx))
            bias = numpy_helper.from_array(hidden_layer.b, name="bias" + str(layer_idx))

            hidden_layer_linear_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=["weight" + str(layer_idx), layer_output_name_list[layer_idx], "bias" + str(layer_idx)],
                outputs=[hidden_linear_output_name]
            )
            model_nodes.append(hidden_layer_linear_node)
            initializer_list.append(weight)
            initializer_list.append(bias)

            # activation
            hidden_activation_output_name = "layer_activation_output" + str(layer_idx)

            # map the operation type
            activation = ""
            if hidden_layer.act == "ReLU":
                activation = "Relu"
            if hidden_layer.act == "sigmoid":
                activation = "Sigmoid"
            if hidden_layer.act == "tanh":
                activation = "Tanh"
            if hidden_layer.act == "Affine":
                activation = "Identity"

            hidden_layer_activation_node = onnx.helper.make_node(
                op_type=activation,
                inputs=[hidden_linear_output_name],
                outputs=[hidden_activation_output_name]
            )

            layer_output_name_list.append(hidden_activation_output_name)
            model_nodes.append(hidden_layer_activation_node)

        # handle scala and offset
        w = numpy_helper.from_array(np.identity(self.num_of_outputs) * self.scale_factor, name="scala_factor")
        b = numpy_helper.from_array(np.ones((self.num_of_outputs, 1)) * self.offset, name="offset")
        output_node = onnx.helper.make_node(
            op_type="Gemm",
            inputs=["scala_factor", layer_output_name_list[-1], "offset"],
            outputs=[model_output]
        )
        model_nodes.append(output_node)
        initializer_list.append(w)
        initializer_list.append(b)

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=model_nodes,
            name="NNController",
            inputs=[state],  # Graph input
            outputs=[control],  # Graph output
            initializer=initializer_list
        )

        onnx_model = onnx.helper.make_model(graph)

        onnx.save(onnx_model, './'+output_file+'.onnx')


# read model from txt file that follows the POLAR NN controller format
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('txt2onnx.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('txt2onnx.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    nn = NeuralNetwork(inputfile)
    nn.write_onnx(outputfile)
    print('Input file is "' + inputfile)
    print('Output file is "' + outputfile)


    # for testing
    onnx_model = onnx.load('./' + outputfile + '.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx_model)
    # [tensor] = [t for t in onnx_model.graph.initializer if t.name == "scala_factor"]
    # tensor = numpy_helper.to_array(tensor)
    # print(tensor)


if __name__ == "__main__":
   main(sys.argv[1:])
