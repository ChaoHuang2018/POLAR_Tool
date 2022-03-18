import yaml

network = 'sig200x200'

nn_verisig = network + '.yml'

dnn_dict = {}
with open(nn_verisig, 'r') as f:
    dnn_dict = yaml.load(f)

layers = len(dnn_dict['activations'])
input_size = len(dnn_dict['weights'][1][0])
output_size = len(dnn_dict['weights'][layers])

for i in dnn_dict['activations']:
    if dnn_dict['activations'][i] == 'Tanh':
        dnn_dict['activations'][i] = 'tanh'
    elif dnn_dict['activations'][i] == 'Sigmoid':
        dnn_dict['activations'][i] = 'sigmoid'
    elif dnn_dict['activations'][i] == 'Linear':
        dnn_dict['activations'][i] = 'Affine'

with open(network, 'w') as nnfile:
    nnfile.write(str(input_size)+"\n")
    nnfile.write(str(output_size)+"\n")
    nnfile.write(str(layers-1)+"\n")    # number of hidden layers
    for i in range(layers-1):           # output size of each hidden layer
        nnfile.write(str(len(dnn_dict['weights'][i+1]))+"\n")
    for i in range(layers):
        nnfile.write(str(dnn_dict['activations'][i+1])+"\n")
    for i in range(layers):
        for j in range(len(dnn_dict['weights'][i+1])):
            for k in range(len(dnn_dict['weights'][i+1][j])):
                nnfile.write(str(dnn_dict['weights'][i+1][j][k])+"\n")
            nnfile.write(str(dnn_dict['offsets'][i+1][j])+"\n")
    nnfile.write(str(0)+"\n")           # output offset
    nnfile.write(str(1)+"\n")           # output scaling factor
