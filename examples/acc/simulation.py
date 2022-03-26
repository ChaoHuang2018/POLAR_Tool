import tensorflow as tf
import numpy as np
import yaml
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

def load_yml_model(network = 'tanh20x20x20'):
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



def simulate_one_step(x, ctrl_model, ctrl_func, ctrl_step):
    simulate_step = 1e-4
    u = ctrl_func(x, ctrl_model)
    steps = int(ctrl_step/simulate_step)
    xs = np.zeros((10, len(x)))
    for i in range(steps):
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + (x[1])*simulate_step
        x_next[1] = x[1] + (x[2])*simulate_step 
        x_next[2] = x[2] + (-2.0*x[2] -0.0001*x[1]*x[1] - 4.0)*simulate_step      # -2.0 * x3 - 0.0001 * x2 * x2 - 4.0
        x_next[3] = x[3] + (x[4])*simulate_step
        x_next[4] = x[4] + (x[5])*simulate_step
        x_next[5] = x[5] + (2.0*u[0] - 2.0*x[5] -0.0001*x[4]*x[4])*simulate_step      # 2.0 * a_ego - 2.0 * x6 - 0.0001 * x5 * x5
        x = x_next
        if x[0] - x[3] - 10.0 - 1.4 * x[4] <= 0.0:  # x1 - x4 - 10.0 - 1.4 * x5 <= 0.0
            print("Crash")
        if (i+1) % (steps//10) == 0:
            xs[(i+1)//(steps//10) - 1] = x
    xs[-1] = x
    return xs, u

def ctrl_input(x, ctrl_model):
    x_t = tf.reshape([30.0, 1.4, x[4], x[0] - x[3], x[1] - x[4]], [1,-1])
    u = ctrl_model(x_t).numpy().flatten()
    return u


def simulation(ctrl_model, n_trajectory = 15, n_steps = 6, ctrl_step = 2e-1, interval = [20, 40]):
    fig, ax = plt.subplots()
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    for _ in range(n_trajectory):
        x0 = np.random.rand(6) * np.array([1, 0.05, 0, 1, 0.05, 0]) + np.array([90, 32, 0, 10, 30, 0])
        print("Initial state:", x0)
        X = simulate_one_trajectory(x0, ctrl_model, ctrl_input, n_steps, ctrl_step)
        for i in range(len(interval)):
            if i+1 < len(interval):
                ax.plot(X[interval[i]:interval[i+1]+1, 0], X[interval[i]:interval[i+1]+1, 2], colors[i%7])
            else:
                ax.plot(X[interval[i]:, 0], X[interval[i]:, 2], colors[i%7])
    # ax.set_xlim([22,33])
    # ax.set_ylim([29.86,30.06])
    plt.show()

def simulate_one_trajectory(x0, model, ctrl_func, n_steps = 6, ctrl_step = 2e-1):
    x = x0
    # x_min = x_max = x
    # u_min = u_max = ctrl_func(x, model)
    X = np.zeros((n_steps*10, 4))
    for i in range(n_steps):
        xs, u = simulate_one_step(x, model, ctrl_func, ctrl_step)
        x = xs[-1]
        # x_min = np.minimum(x_min, x)
        # x_max = np.maximum(x_max, x)
        # u_min = np.minimum(u_min, u)
        # u_max = np.maximum(u_max, u)
        X[i*10:(i+1)*10, 0:2] = xs[:, 1:3]
        X[i*10:(i+1)*10, 2:4] = xs[:, 4:6]
    # print("x_min =", x_min, "\nx_max =", x_max)
    # print("u_min =", u_min, "\nu_max =", u_max)
    return X

if __name__ == '__main__':
    tf_model, input_size = load_yml_model(network='tanh20x20x20')
    simulation(tf_model, n_trajectory=10, n_steps=50, ctrl_step=1e-1, interval=[0, 100, 200, 300, 400])
