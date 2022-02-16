import numpy as np
from neuralnetwork import NN
from tensorflow.keras.models import model_from_json


def nn_controller(filename, dir):
    """
    Return the network controller function
    """
    # load json and create model
    json_filename = dir + filename + '.json'
    json_file = open(dir + filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(dir + filename + '.h5')
    print("Loaded kera model from disk.")
    NN_controller = NN(
        name=filename,
        keras=True,
        model=loaded_model,
        model_json=json_filename
    )
    return NN_controller
