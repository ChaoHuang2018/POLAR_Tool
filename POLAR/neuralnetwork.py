import json
import re
import os,sys
import numpy as np
from numpy import linalg as LA
from keras import backend as K


class NN(object):
    """
    a neural network with relu activation function
    """
    def __init__(
        self,
        name=None,
        res=None,
        activation=None,
        keras=False,
        model=None,
        model_json=None
    ):
        self.name = name
        self.eran = False
        self.mean = 0.0
        self.std = 1.0
        if not keras:
            # activation type
            activations = activation.split('_')
            if len(activations) > 1:
                self.activation = activations[0]
                self.last_layer_activation = activations[1]
            else:
                self.activation = activation
                self.last_layer_activation = None
            # affine mapping of the output
            self.offset = res[-2]
            self.scale_factor = res[-1]

            # parse structure of neural networks
            self.num_of_inputs = int(res[0])
            self.num_of_outputs = int(res[1])
            self.num_of_hidden_layers = int(res[2])
            self.network_structure = np.zeros(self.num_of_hidden_layers + 1,
                                              dtype=int)

            self.activations = ([self.activation] *
                                (self.num_of_hidden_layers + 1))
            if self.last_layer_activation is not None:
                self.activations[-1] = self.last_layer_activation

            # pointer is current reading index
            self.pointer = 3

            # num of neurons of each layer
            for i in range(self.num_of_hidden_layers):
                self.network_structure[i] = int(res[self.pointer])
                self.pointer += 1

            # output layer
            self.network_structure[-1] = self.num_of_outputs

            # all values from the text file
            self.param = res

            # store the weights and bias in two lists
            # self.weights
            # self.bias
            self.parse_w_b()
        elif keras == 'eran':
            self.eran = True
            self.type = None
            self.layers = None
            params = []
            self.weights = []
            self.bias = []
            self.model = model
            self.config = open(model, 'r')
            self.set_type()
            self.set_layer()
            print('Layer type: {}'.format(self.layers[0].type))
        else:
            self.type = None
            self.layers = None
            params = []
            self.weights = []
            self.bias = []
            self.model = model
            self.model_json = model_json
            with open(model_json) as json_file:
                self.config = json.load(json_file)
            self.set_type()
            self.set_layer()
            for layer in model.layers:
                params.append(layer.get_weights())  # list of numpy arrays
            for param in params:
                if len(param) == 0:
                    continue
                else:
                    self.weights.append(param[0])
                    self.bias.append(param[1])

    def set_type(self):
        if self.config and not self.eran:
            for class_name in self.config['config']['layers']:
                if class_name['class_name'][:4] == 'Conv':
                    self.type = 'Convolutional'
                elif class_name['class_name'] == 'Flatten':
                    self.type = 'Flatten'
            if not self.type:
                self.type = 'Fully_connected'

    def set_layer(self):
        self.layers = []
        if not self.eran:
            with open(self.model_json[:-5] + '.pyt', "w") as eran_model:
                eran_model.write('Normalize mean=[0] std=[1]')
                eran_model.write('\n')
                layers_config = self.config['config']['layers']
                for idx, layer in enumerate(self.model.layers):
                    layer_tmp = Layer()
                    layer_activation = None
                    layer_config = layers_config[idx]
                    layer_detail = layer_config['config']
                    if layer_config['class_name'] == 'Flatten':
                        layer_tmp._type = 'Flatten'
                        flatten_input_shape = layer.input_shape[1:]
                        if len(flatten_input_shape) == 2:
                            flatten_input_shape = list(flatten_input_shape)
                            flatten_input_shape.append(1)
                            flatten_input_shape = tuple(flatten_input_shape)
                        else:
                            flatten_input_shape = layer.input_shape[1:]
                        layer_tmp._input_dim = flatten_input_shape
                        layer_tmp._output_dim = layer.output_shape[1:]
                        self.layers.append(layer_tmp)
                    elif layer_config['class_name'] == 'Conv2D':
                        layer_tmp._type = 'Convolutional'
                        layer_tmp._input_dim = layer.input_shape[1:]
                        layer_tmp._output_dim = layer.output_shape[1:]
                        layer_tmp._kernel = layer.get_weights()[0]
                        layer_tmp._bias = layer.get_weights()[1]
                        layer_tmp._stride = layer.strides
                        layer_tmp._activation = self.activation_function(
                            layer_detail['activation']
                        )
                        layer_tmp._filter_size = layer.filters
                        if layer.padding == 'valid':
                            layer_tmp._padding = 0
                        elif layer.padding == 'same':
                            layer_tmp._padding =(
                                layer_tmp.kernel.shape[0] -
                                layer_tmp.stride[0]
                            )/2
                        self.layers.append(layer_tmp)

                        # Activation layer
                        layer_activation = Layer()
                        layer_activation._type = 'Activation'
                        layer_activation._activation = layer_tmp.activation
                        layer_activation._input_dim, layer_activation._output_dim = (
                            layer_tmp.output_dim,
                            layer_tmp.output_dim
                        )
                        self.layers.append(layer_activation)
                        eran_model.write('Conv2D')
                        eran_model.write('\n')
                        eran_model.write(
                            self.eran_activation(layer_tmp.activation) +
                            ', filters=' + str(layer_tmp.output_dim[2]) +
                            ', kernel_size=' + str(
                                list(layer_tmp.kernel.shape[:2])
                            ) +
                            ', input_shape=' + str(
                                list(layer_tmp.input_dim)
                            ) +
                            ', stride=' + str(list(layer_tmp.stride)) +
                            ', padding=' + str(layer_tmp.padding)
                        )
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.kernel.tolist()))
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.bias.tolist()))
                        eran_model.write('\n')

                    elif (
                        layer_config['class_name'] == 'Dense' and
                        layer_detail['activation'] != 'softmax'
                    ):
                        layer_tmp._type = 'Fully_connected'
                        layer_tmp._input_dim = layer.output_shape[1:]
                        layer_tmp._output_dim = layer.output_shape[1:]
                        layer_tmp._activation = self.activation_function(
                            layer_detail['activation']
                        )
                        params = layer.get_weights()
                        layer_tmp._weight = params[0]
                        layer_tmp._bias = params[1]
                        self.layers.append(layer_tmp)
                        eran_model.write(
                            self.eran_activation(layer_tmp.activation)
                        )
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.weight.T.tolist()))
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.bias.tolist()))
                        eran_model.write('\n')

                    elif (
                        layer_config['class_name'] == 'Dense' and
                        layer_detail['activation'] == 'softmax'
                    ):
                        layer_tmp._type = 'Fully_connected'
                        layer_tmp._input_dim = layer.output_shape[1:]
                        layer_tmp._output_dim = layer.output_shape[1:]
                        layer_tmp._activation = self.activation_function(
                            layer_detail['activation']
                        )
                        params = layer.get_weights()
                        layer_tmp._weight = params[0]
                        layer_tmp._bias = params[1]
                        self.layers.append(layer_tmp)
                        eran_model.write(
                            self.eran_activation(layer_tmp.activation)
                        )
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.weight.T.tolist()))
                        eran_model.write('\n')
                        eran_model.write(str(layer_tmp.bias.tolist()))
                        eran_model.write('\n')

                    elif layer_config['class_name'] == 'MaxPooling2D':
                        layer_tmp._type = 'Pooling'
                        layer_tmp._activation = 'max'
                        layer_tmp._stride = layer.strides
                        layer_tmp._filter_size = layer.pool_size
                        layer_tmp._input_dim = layer.input_shape[1:]
                        layer_tmp._output_dim = layer.output_shape[1:]
                        self.layers.append(layer_tmp)

                    elif layer_config['class_name'] == 'AveragePooling2D':
                        layer_tmp._type = 'Pooling'
                        layer_tmp._activation = 'average'
                        layer_tmp._stride = layer.strides
                        layer_tmp._filter_size = layer.pool_size
                        layer_tmp._input_dim = layer.input_shape[1:]
                        layer_tmp._output_dim = layer.output_shape[1:]
                        self.layers.append(layer_tmp)
        else:
            self.type = 'Flatten'

            last_line = None

            while True:
                curr_line = self.config.readline()[:-1]
                if 'Normalize' in curr_line:
                    self.mean = extract_mean(curr_line)
                    self.std = extract_std(curr_line)
                elif curr_line in ["ReLU", "Sigmoid", "Tanh", "Affine"]:
                    if last_line in ["Conv2D"]:
                        layer_tmp = Layer()
                        layer_tmp._type = 'Flatten'
                        flatten_input_shape = self.layers[-1].input_dim
                        if len(flatten_input_shape) == 2:
                            flatten_input_shape = list(flatten_input_shape)
                            flatten_input_shape.append(1)
                            flatten_input_shape = tuple(flatten_input_shape)
                        layer_tmp._input_dim = flatten_input_shape
                        layer_tmp._output_dim = (np.prod(flatten_input_shape),
                                                 1)
                        self.layers.append(layer_tmp)

                    layer_tmp = Layer()
                    W = self.parseVec(self.config).T
                    b = self.parseVec(self.config).T
                    self.weights.append(W.T)
                    self.bias.append(b.reshape(-1, 1))
                    layer_tmp._type = 'Fully_connected'
                    layer_tmp._input_dim = [W.shape[1]]
                    layer_tmp._output_dim = [W.shape[1]]
                    if (curr_line == 'ReLU'):
                        layer_tmp._activation = 'ReLU'
                        self.activation = 'ReLU'
                        self.last_layer_activation = 'ReLU'
                    elif (curr_line == 'Sigmoid'):
                        layer_tmp._activation = 'sigmoid'
                        self.activation = 'sigmoid'
                        self.last_layer_activation = 'Affine'
                    elif (curr_line == 'Tanh'):
                        layer_tmp._activation = 'tanh'
                        self.activation = 'tanh'
                        self.last_layer_activation = 'Affine'
                    elif (curr_line == 'Affine'):
                        layer_tmp._activation = 'Affine'
                    layer_tmp._weight = W
                    layer_tmp._bias = b
                    self.layers.append(layer_tmp)
                elif curr_line == "Conv2D":
                    self.type = 'Convolutional'
                    line = self.config.readline()
                    layer_tmp = Layer()
                    start = 0
                    if("ReLU" in line):
                        start = 5
                        layer_tmp._activation = "ReLU"
                    elif("Sigmoid" in line):
                        start = 8
                        layer_tmp._activation = "sigmoid"
                    elif("Tanh" in line):
                        start = 5
                        layer_tmp._activation = "tanh"
                    elif("Affine" in line):
                        start = 7
                        layer_tmp._activation = "Affine"
                    if 'padding' in line:
                        args = runRepl(line[start:-1], [
                            "filters", "input_shape", "kernel_size",
                            "stride", "padding"
                        ])
                        output_height = (
                            args["input_shape"][0] -
                            args["kernel_size"][0] +
                            2 * args["padding"]
                        ) / args["stride"][0] + 1
                        output_width = (
                            args["input_shape"][1] -
                            args["kernel_size"][1] +
                            2 * args["padding"]
                        ) / args["stride"][1] + 1
                        layer_tmp._padding = args["padding"]
                    else:
                        args = runRepl(line[start:-1], [
                            "filters", "input_shape", "kernel_size"
                        ])
                        output_height = (
                            args["input_shape"][0] -
                            args["kernel_size"][0]
                        ) / args["stride"][0] + 1
                        output_width = (
                            args["input_shape"][1] -
                            args["kernel_size"][1]
                        ) / args["stride"][1] + 1

                    W = self.parseVec(self.config)
                    b = self.parseVec(self.config)
                    layer_tmp._type = 'Convolutional'
                    layer_tmp._input_dim = tuple(args["input_shape"])
                    layer_tmp._output_dim = tuple([
                        output_height, output_width, args["filters"]
                    ])
                    layer_tmp._kernel = W
                    layer_tmp._bias = b
                    layer_tmp._stride = args["stride"]
                    layer_tmp._filter_size = args["filters"]
                    self.layers.append(layer_tmp)

                    # Activation layer
                    layer_activation = Layer()
                    layer_activation._type = 'Activation'
                    layer_activation._activation = layer_tmp.activation
                    layer_activation._input_dim, layer_activation._output_dim = (
                        layer_tmp.output_dim,
                        layer_tmp.output_dim
                    )
                    self.layers.append(layer_activation)

                elif curr_line == "":
                    break
                last_line = curr_line

            if self.type == 'Flatten':
                layer_tmp = Layer()
                layer_tmp._type = 'Flatten'
                image_size = 28
                flatten_input_shape = (image_size, image_size)
                if len(flatten_input_shape) == 2:
                    flatten_input_shape = list(flatten_input_shape)
                    flatten_input_shape.append(1)
                    flatten_input_shape = tuple(flatten_input_shape)
                else:
                    flatten_input_shape = self.weights[0].shape[0]
                layer_tmp._input_dim = flatten_input_shape
                layer_tmp._output_dim = [self.weights[0].shape[1]]
                self.layers = [layer_tmp] + self.layers
            print(self.layers[0].output_dim)

    def activation_function(self, activation_type):
        if activation_type == 'relu':
            return 'ReLU'
        elif activation_type == 'softmax':
            return 'Affine'
        else:
            return activation_type

    def eran_activation(self, activation_type):
        if activation_type == 'relu':
            return 'ReLU'
        elif activation_type == 'softmax':
            return 'Affine'
        elif activation_type == 'sigmoid':
            return 'Sigmoid'
        elif activation_type == 'tanh':
            return 'Tanh'
        else:
            return activation_type

    def keras_model(self, x):
        if self.model is not None:
            y = self.model.predict(x)
            return y

    def keras_model_pre_softmax(self, x):
        get_output_pre_softmax = K.function([self.model.layers[0].input],
                                            [self.model.layers[0].output])
        layer_output = get_output_pre_softmax([x])[0][0][0][0]
        return layer_output

    def activate(self, x):
        """
        activation function
        """
        if self.activation == 'ReLU':
            x[x < 0] = 0
        elif self.activation == 'tanh':
            x = np.tanh(x)
        elif self.activation == 'sigmoid':
            x = 1/(1 + np.exp(-x))
        elif self.activation == 'Affine':
            return x
        return x

    def last_layer_activate(self, x):
        """
        activation function
        """
        if self.last_layer_activation == 'ReLU':
            x[x < 0] = 0
        elif self.last_layer_activation == 'tanh':
            x = np.tanh(x)
        elif self.last_layer_activation == 'sigmoid':
            x = 1/(1 + np.exp(-x))
        elif self.activation == 'Affine':
            return x
        return x

    def parse_w_b(self):
        """
        Parse the input text file
        and store the weights and bias indexed by layer
        Generate: self.weights, self.bias
        """
        # initialize the weights and bias storage space
        self.weights = [None] * (self.num_of_hidden_layers + 1)
        self.bias = [None] * (self.num_of_hidden_layers + 1)

        # compute parameters of the input layer
        weight_matrix0 = np.zeros((self.network_structure[0],
                                   self.num_of_inputs), dtype=np.float64)
        bias_0 = np.zeros((self.network_structure[0], 1), dtype=np.float64)

        for i in range(self.network_structure[0]):
            for j in range(self.num_of_inputs):
                weight_matrix0[i, j] = self.param[self.pointer]
                self.pointer += 1

            bias_0[i] = self.param[self.pointer]
            self.pointer += 1

        # store input layer parameters
        self.weights[0] = weight_matrix0
        self.bias[0] = bias_0

        # compute the hidden layers paramters
        for i in range(self.num_of_hidden_layers):
            weights = np.zeros((self.network_structure[i + 1],
                                self.network_structure[i]), dtype=np.float64)
            bias = np.zeros((self.network_structure[i + 1], 1),
                            dtype=np.float64)

            # read the weight matrix
            for j in range(self.network_structure[i + 1]):
                for k in range(self.network_structure[i]):
                    weights[j][k] = self.param[self.pointer]
                    self.pointer += 1
                bias[j] = self.param[self.pointer]
                self.pointer += 1

            # store parameters of each layer
            self.weights[i + 1] = weights
            self.bias[i + 1] = bias

    def controller(self, x):
        """
        Input: state
        Output: control value after affine transformation
        """
        # transform the input
        g = x.reshape([-1, 1])

        # pass input through each layer
        for i in range(self.num_of_hidden_layers - 2):
            # linear transformation
            print(self.weights[i].shape)
            g = self.weights[i] @ g
            g = g + self.bias[i]

            # activation
            g = self.activate(g)

        # output layer
        if self.last_layer_activation is not None:
            # linear transformation
            g = self.weights[self.num_of_hidden_layers - 2] @ g
            g = g + self.bias[self.num_of_hidden_layers - 2]

            # activation
            g = self.last_layer_activate(g)
        else:
            # linear transformation
            g = self.weights[self.num_of_hidden_layers - 2] @ g
            g = g + self.bias[self.num_of_hidden_layers - 2]

            # activation
            g = self.activate(g)

        # affine transformation of output
        # y = g - self.offset
        # y = y * self.scale_factor
        y = g

        return y

    @property
    def lips(self):
        if self.activation == 'ReLU':
            scalar = 1
        elif self.activation == 'tanh':
            scalar = 1
        elif self.activation == 'sigmoid':
            scalar = 1/4
        # initialize L cosntant
        L = 1.0
        # multiply norm of weights in each layer
        for i, weight in enumerate(self.weights):
            L *= scalar * LA.norm(weight, 2)

        # activation function of output layer is not the same as other layers
        if self.last_layer_activation is not None:
            if self.activation == 'ReLU':
                L *= 1
            elif self.activation == 'tanh':
                L *= 1
            elif self.activation == 'sigmoid':
                L *= 1/4

        return (L - self.offset) * self.scale_factor

    @property
    def num_of_hidden_layers(self):
        return len(self.layers)

    def parseVec(self, net):
        return np.array(eval(net.readline()[:-1]))
        
        
    def print_nn(self):
        if (self.type == 'Fully_connected'):
            with open(self.name, 'w') as nnfile:
                nnfile.write(str(self.weights[0].shape[0])+"\n")
                nnfile.write(str(self.weights[-1].shape[1])+"\n")
                nnfile.write(str(len(self.weights))+"\n")
                for i in range(len(self.weights)-1):
                    nnfile.write(str(self.weights[i].shape[1])+"\n")
                for i in range(len(self.weights)-1):
                    nnfile.write(str(self.weights[i].shape[1])+"\n")
                for i in range(len(self.weights)):
                    nnfile.write(str(self.layers[i]._activation)+"\n")
                for i in range(len(self.weights)):
                    self.layers[i].print_layer(nnfile)
                nnfile.write(str(0)+"\n")
                nnfile.write(str(1)+"\n")
                

class Layer(object):
    """
    Layer class with following properties:
        type
        weight
        bias
        kernel
        padding
        stride
        activation
        filter_size
        input_dim
        output_dim
        layer_idx
    """
    def __init__(self):
        self._type = None
        self._weight = None
        self._bias = None
        self._kernel = None
        self._padding = None
        self._stride = None
        self._activation = None
        self._filter_size = None
        self._input_dim = None
        self._output_dim = None
        self._layer_idx = None

    @property
    def type(self):
        return self._type

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    @property
    def kernel(self):
        return self._kernel

    @property
    def padding(self):
        return self._padding

    @property
    def stride(self):
        return self._stride

    @property
    def activation(self):
        return self._activation

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def input_dim(self):
        return tuple(int(ele) for ele in self._input_dim)

    @property
    def output_dim(self):
        return tuple(int(ele) for ele in self._output_dim)
        
    def print_layer(self, nnfile):
        if (self._type = 'Fully_connected'):
            for i in range(self._weight.shape[1]):
                for j in range(self._weight.shape[0])
                    nnfile.write(str(self._weight[i][j])+"\n")
                nnfile.write(str(self._bias[i]))


def extract_mean(text):
    m = re.search('mean=\[(.+?)\]', text)

    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
        mean_array[i] = np.float64(mean_str[i])
    return mean_array


def extract_std(text):
    m = re.search('std=\[(.+?)\]', text)
    if m:
        stds = m.group(1)
    std_str =stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array


def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")
