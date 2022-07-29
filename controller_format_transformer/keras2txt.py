import argparse

from tensorflow.keras.models import load_model


def keras2txt(h5_file, txt_file):
    # load keras model from the h5 file
    model = load_model(h5_file)
    print("============keras model summary============")
    print(model.summary())

    # get the input dim
    input_dim = model.layers[0].get_config()['batch_input_shape'][1]
    # get the output dim
    output_dim = model.layers[-1].get_config()['units']
    # var to record number of hidden neurons
    num_of_hidden_neurons = []
    # var to record types of activation function
    activations = []

    for _layer in model.layers[1:]:
        layer_config = _layer.get_config()
        if 'layers' in layer_config:
            # if there is a functional model in a layer, this layer actually
            # contains layers that cannot be read from the model directly.
            for _layer_in_model in _layer.layers[1:]:
                layer_config = _layer_in_model.get_config()
                # append the current layer's number of neurons
                num_of_hidden_neurons.append(layer_config['units'])
                # append the current layer's activation function type
                activations.append(layer_config['activation'])
        else:
            # append the current layer's number of neurons
            num_of_hidden_neurons.append(layer_config['units'])
            # append the current layer's activation function type
            activations.append(layer_config['activation'])

    # get the number of hidden layers
    num_of_hidden_layer = len(activations) - 1

    with open(txt_file, 'w') as output_file:
        # write the neural network architecture
        output_file.write('{}'.format(input_dim) + '\n')
        output_file.write('{}'.format(output_dim) + '\n')
        output_file.write('{}'.format(num_of_hidden_layer) + '\n')

        for _num_neurons in num_of_hidden_neurons[:-1]:
            output_file.write('{}'.format(_num_neurons) + '\n')

        for _activation in activations:
            if _activation == 'linear':
                _activation = 'Affine'
            output_file.write('{}'.format(_activation) + '\n')

        # write weights and biases
        for _layer in model.layers[1:]:
            layer_config = _layer.get_config()
            if 'layers' in layer_config:
                # if there is a functional model in a layer, this layer
                # actually contains layers that cannot be read from the model
                # directly.
                for _layer_in_model in _layer.layers[1:]:
                    weights, biases = _layer_in_model.get_weights()
                    # wrtie weights
                    for _col in range(weights.shape[1]):
                        for _row in range(weights.shape[0]):
                            output_file.write(
                                '{}'.format(weights[_row, _col]) + '\n'
                            )
                    # write biases
                    for _idx_neuron in range(biases.shape[0]):
                        output_file.write('{}'.format(
                            biases[_idx_neuron]) + '\n'
                        )

            else:
                weights, biases = _layer.get_weights()
                # wrtie weights
                for _col in range(weights.shape[1]):
                    for _row in range(weights.shape[0]):
                        output_file.write('{}'.format(
                            weights[_row, _col]) + '\n'
                        )
                # write biases
                for _idx_neuron in range(biases.shape[0]):
                    output_file.write('{}'.format(biases[_idx_neuron]) + '\n')

        # write default scalar and offset
        output_file.write('{}'.format(0) + '\n')
        output_file.write('{}'.format(1) + '\n')
    print("============{} saved============".format(txt_file))
    print("============Done============")


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--keras_model_file', type=str, help='path to the h5 model file'
    )
    parser.add_argument(
        '--output_txt_model_file', type=str,
        help='path to the output txt model file'
    )
    args = parser.parse_args()

    keras2txt(args.keras_model_file, args.output_txt_model_file)
