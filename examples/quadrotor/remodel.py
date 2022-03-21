import tensorflow as tf
import numpy as np
import yaml
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

def load_yml_model(network = 'tanh20x20'):
    nn_verisig = network + '.yml'
    dnn_dict = {}
    with open(nn_verisig, 'r') as f:
        dnn_dict = yaml.load(f)

    layers = len(dnn_dict['activations'])
    input_size = len(dnn_dict['weights'][1][0])
    # output_size = len(dnn_dict['weights'][layers])

    tf_model = tf.keras.Sequential()
    for i in range(1, layers+1):
        input_shape = (len(dnn_dict['weights'][i][0]),)
        output_shape = len(dnn_dict['weights'][i])
        if dnn_dict['activations'][i] == 'Tanh':
            dnn_dict['activations'][i] = 'tanh'
        elif dnn_dict['activations'][i] == 'Sigmoid':
            dnn_dict['activations'][i] = 'sigmoid'
        elif dnn_dict['activations'][i] == 'Relu':
            dnn_dict['activations'][i] = 'relu'
        else:
            dnn_dict['activations'][i] = 'linear'
        layer = tf.keras.layers.Dense(output_shape, input_shape=input_shape, activation=dnn_dict['activations'][i])
        layer(tf.zeros((1, input_shape[0])))
        layer.set_weights([np.transpose(dnn_dict['weights'][i]), np.array(dnn_dict['offsets'][i])])
        tf_model.add(layer)
    return tf_model, input_size

def convert_model(discrete_model, n_output = 3):
    continuous_model = tf.keras.Sequential()
    for layer in discrete_model.layers[:-1]:
        continuous_model.add(tf.keras.layers.Dense(layer.units, activation=layer.activation))
    last_hidden = discrete_model.layers[-1]
    continuous_model.add(tf.keras.layers.Dense(n_output, activation='tanh'))      # One more layer
    continuous_model.add(tf.keras.layers.Dense(n_output))
    return continuous_model
    
def distill_model(old_model, new_model, input_shape, epochs = 100_000, batch_size = 512, learning_rate=1e-3):
    table = tf.constant([
        [-0.1, -0.1, 7.81],
        [-0.1, -0.1, 11.81],
        [-0.1, 0.1, 7.81],
        [-0.1, 0.1, 11.81],
        [0.1, -0.1, 7.81],
        [0.1, -0.1, 11.81],
        [0.1, 0.1, 7.81],
        [0.1, 0.1, 11.81],
        ])
    def _output(out):
        idxs = tf.math.argmax(out, 1)
        return tf.gather(table, idxs)
    
    stds = tf.constant([0.5, 0.5, 0.2, 1.0, 1.0, 0.5])
    scale = tf.constant([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, epochs/20, 1.0) # 20 times decay eventually

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    for epoch in range(epochs):
        inputs = tf.random.normal(shape=[batch_size, input_shape], mean=0, stddev=stds)
        labels = _output(old_model(tf.math.multiply(inputs, scale), training=False))
        with tf.GradientTape() as tape:
            logits = new_model(inputs, training=True)
            loss_value = loss_fn(labels, logits) + tf.reduce_sum(new_model.losses)
        grads = tape.gradient(loss_value, new_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, new_model.trainable_weights))
        if epoch % (epochs // 100) == 0:
            print(f"Epoch {epoch}, loss = {loss_value.numpy()}")
    
    return new_model

def convert_classify_model(discrete_model, n_output = 3):
    classify_model = tf.keras.Sequential()
    for layer in discrete_model.layers[:-1]:
        classify_model.add(tf.keras.layers.Dense(layer.units, activation=layer.activation))
    classify_model.add(tf.keras.layers.Dense(n_output, activation='sigmoid'))
    return classify_model

def distill_classify_model(old_model, new_model, input_shape, epochs = 100_000, batch_size = 512, learning_rate=1e-3):
    table = tf.constant([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        ])
    def _output(out):
        idxs = tf.math.argmax(out, 1)
        return tf.gather(table, idxs)
    
    stds = tf.constant([0.5, 0.5, 0.2, 1.0, 1.0, 0.5])
    scale = tf.constant([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    
    learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, epochs/20, 1.0) # 20 times decay eventually

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    for epoch in range(epochs):
        inputs = tf.random.normal(shape=[batch_size, input_shape], mean=0, stddev=stds)
        labels = _output(old_model(tf.math.multiply(inputs, scale), training=False))
        with tf.GradientTape() as tape:
            logits = new_model(inputs, training=True)
            loss_value = loss_fn(labels, logits)
        grads = tape.gradient(loss_value, new_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, new_model.trainable_weights))
        if epoch % (epochs // 100) == 0:
            print(f"Epoch {epoch}, loss = {loss_value.numpy()}")
    
    return new_model

def append_output_layer(model):
    x0 = tf.zeros((1,6))
    x = model(x0, training=False)
    out_layer = tf.keras.layers.Dense(3)
    _ = out_layer(x)
    kernel = tf.linalg.diag([0.2, 0.2, 4.0])
    bias = tf.constant([-0.1, -0.1, 7.81])
    out_layer.set_weights([kernel, bias])
    model.add(out_layer)
    return model


def save_polar_model(model, input_size, network_name = 'tanh20x20'):
    layers = len(model.layers)
    output_size = 3
    network_name = network_name + '_remodel'
    with open(network_name, 'w') as nnfile:
        nnfile.write(str(input_size)+"\n")
        nnfile.write(str(output_size)+"\n")
        nnfile.write(str(layers - 1)+"\n")      # number of hidden layers
        for layer in model.layers[:-1]:         # output size of each hidden layer
            nnfile.write(str(layer.kernel.shape[1])+"\n")
        for layer in model.layers:
            activation = layer.activation.__name__
            if activation == 'relu':
                activation = 'ReLU'
            elif activation == 'linear':
                activation = 'Affine'
            nnfile.write(str(activation)+"\n")
        for layer in model.layers:
            for j in range(layer.kernel.shape[1]):
                for k in range(layer.kernel.shape[0]):
                    nnfile.write(str(layer.kernel[k][j].numpy())+"\n")
                nnfile.write(str(layer.bias[j].numpy())+"\n")
        nnfile.write(str(0)+"\n")               # output offset
        nnfile.write(str(1)+"\n")               # output scaling factor
    print(f"Polar model {network_name} saved.")

def simulate_one_step(x, ctrl_model, ctrl_func, ctrl_step):
    simulate_step = 1e-3
    u = ctrl_func(x, ctrl_model)
    steps = int(ctrl_step/simulate_step)
    xs = np.zeros((10, len(x)))
    for i in range(steps):
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + (x[3] - 0.25)*simulate_step
        x_next[1] = x[1] + (x[4] + 0.25)*simulate_step
        x_next[2] = x[2] + (x[5])*simulate_step
        x_next[3] = x[3] + (9.81*math.sin(u[0])/math.cos(u[0]))*simulate_step
        x_next[4] = x[4] + (-9.81*math.sin(u[1])/math.cos(u[1]))*simulate_step
        x_next[5] = x[5] + (-9.81+u[2])*simulate_step
        x = x_next
        if (i+1) % (steps//10) == 0:
            xs[(i+1)//(steps//10) - 1] = x
    xs[-1] = x
    return xs, u

def ctrl_input(x, ctrl_model):
    x_t = tf.reshape(x, [1,-1])
    u = ctrl_model(x_t).numpy().flatten()
    return u

def bangbang_ctrl(x, bangbang_model):
    table = tf.constant([
        [-0.1, -0.1, 7.81],
        [-0.1, -0.1, 11.81],
        [-0.1, 0.1, 7.81],
        [-0.1, 0.1, 11.81],
        [0.1, -0.1, 7.81],
        [0.1, -0.1, 11.81],
        [0.1, 0.1, 7.81],
        [0.1, 0.1, 11.81],
        ])
    scale = tf.constant([0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=tf.double)
    def _output(out):
        idxs = tf.math.argmax(out, 1)
        return tf.gather(table, idxs)
    
    x_t = tf.reshape(x, [1,-1])
    u = _output(bangbang_model(tf.math.multiply(x_t, scale))).numpy().flatten()
    return u


def simulation(ctrl_model, bangbang_model, n_trajectory = 15):
    fig, axs = plt.subplots(ncols=2)
    for _ in range(n_trajectory):
        x0 = np.random.rand(6) * np.array([0.025, 0.025, 0, 0, 0, 0]) + np.array([-0.05, -0.025, 0, 0, 0, 0])
        print("Initial state:", x0[:2])
        X_new = simulate_one_trajectory(x0, ctrl_model, ctrl_input)
        X_bangbang = simulate_one_trajectory(x0, bangbang_model, bangbang_ctrl)
        interval = [20, 40]
        axs[0].plot(X_new[:interval[0]+1, 0], X_new[:interval[0]+1, 1], 'lightsteelblue')
        axs[0].plot(X_new[interval[0]:interval[1]+1, 0], X_new[interval[0]:interval[1]+1, 1], 'royalblue')
        axs[0].plot(X_new[interval[1]:, 0], X_new[interval[1]:, 1], 'navy')
        axs[1].plot(X_bangbang[:interval[0]+1, 0], X_bangbang[:interval[0]+1, 1], 'mistyrose')
        axs[1].plot(X_bangbang[interval[0]:interval[1]+1, 0], X_bangbang[interval[0]:interval[1]+1, 1], 'red')
        axs[1].plot(X_bangbang[interval[1]:, 0], X_bangbang[interval[1]:, 1], 'darkred')
    axs[0].set_title("Remodelled model")
    axs[1].set_title("Verisig model")
    plt.show()

def simulate_one_trajectory(x0, model, ctrl_func, n_steps = 6, ctrl_step = 2e-1):
    x = x0
    x_min = x_max = x
    u_min = u_max = ctrl_func(x, model)
    X = np.zeros((n_steps*10, 2))
    for i in range(n_steps):
        xs, u = simulate_one_step(x, model, ctrl_func, ctrl_step)
        x = xs[-1]
        x_min = np.minimum(x_min, x)
        x_max = np.maximum(x_max, x)
        u_min = np.minimum(u_min, u)
        u_max = np.maximum(u_max, u)
        X[i*10:(i+1)*10] = xs[:, :2]
    print("x_min =", x_min, "\nx_max =", x_max)
    print("u_min =", u_min, "\nu_max =", u_max)
    return X

def save_json_model(dnn_model,  network_name = 'tanh20x20'):
    # serialize model to JSON
    model_json = dnn_model.to_json()
    with open("./model/"+network_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    dnn_model.save_weights("./model/"+network_name+".h5")
    print("Saved model to disk")

def load_json_model(network_name = 'tanh20x20'):
    json_filename = './model/' + network_name + '.json'
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dnn_model = model_from_json(loaded_model_json)
    dnn_model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    # load weights into new model
    dnn_model.load_weights('./model/' + network_name + '.h5')
    return dnn_model

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    tf_model, input_size = load_yml_model()
    # new_model = convert_classify_model(tf_model)
    # new_model = distill_classify_model(tf_model, new_model, input_size)
    # new_model = append_output_layer(new_model)
    new_model = load_json_model()
    simulation(new_model, tf_model, n_trajectory=30)
    # save_polar_model(new_model, input_size)
    # save_json_model(new_model)
