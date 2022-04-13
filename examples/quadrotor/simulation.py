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



def simulate_one_step(x, ctrl_model, ctrl_func, ctrl_step, iter):
    target_v = [-0.25, 0.25]
    if iter  >= 10:
        target_v = [-0.25, -0.25]
    if iter >= 20:
        target_v = [0.0, 0.25]
    if iter >= 25:
        target_v = [0.25, -0.25]

    simulate_step = 5e-5
    u = ctrl_func(x, ctrl_model)
    steps = int(ctrl_step/simulate_step)
    xs = np.zeros((10, len(x)))
    for i in range(steps):
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + (x[3] + target_v[0])*simulate_step
        x_next[1] = x[1] + (x[4] + target_v[1])*simulate_step
        x_next[2] = x[2] + (x[5])*simulate_step
        x_next[3] = x[3] + (9.81*math.sin(u[0])/math.cos(u[0]))*simulate_step
        x_next[4] = x[4] + (-9.81*math.sin(u[1])/math.cos(u[1]))*simulate_step
        x_next[5] = x[5] + (-9.81+u[2])*simulate_step
        x = x_next
        if (i+1) % (steps//10) == 0:
            xs[(i+1)//(steps//10) - 1] = x
    xs[-1] = x
    return xs, u

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


def simulation(ctrl_model, n_trajectory = 15, n_steps = 6, ctrl_step = 2e-1, interval = [20, 40]):
    fig, axs = plt.subplots(nrows=2,ncols=2)
    colors = ['k', 'g', 'r', 'c', 'm', 'y']
    # polar_res = read_gnuplot_file(filename = "autosig.plt")
    polar_res = read_gnuplot_file(filename = "polar_quadrotor_verisig_30_steps_x_y_1_0.05.plt")
    for octagon in polar_res:
        axs[0,0].plot(octagon[:, 0], octagon[:, 1], 'b', linewidth=0.2)
    polar_res = read_gnuplot_file(filename = "polar_quadrotor_verisig_30_steps_x_vx_1_0.05.plt")
    for octagon in polar_res:
        axs[0,1].plot(octagon[:, 0], octagon[:, 1], 'b', linewidth=0.2)
    polar_res = read_gnuplot_file(filename = "polar_quadrotor_verisig_30_steps_y_vy_1_0.05.plt")
    for octagon in polar_res:
        axs[1,0].plot(octagon[:, 0], octagon[:, 1], 'b', linewidth=0.2)
    polar_res = read_gnuplot_file(filename = "polar_quadrotor_verisig_30_steps_z_vz_1_0.05.plt")
    for octagon in polar_res:
        axs[1,1].plot(octagon[:, 0], octagon[:, 1], 'b', linewidth=0.2)
    for _ in range(n_trajectory):
        x0 = np.random.rand(6) * np.array([0.025, 0.025, 0, 0, 0, 0]) + np.array([-0.05, -0.025, 0, 0, 0, 0])
        print("Initial state:", x0)
        X = simulate_one_trajectory(x0, ctrl_model, bangbang_ctrl, n_steps, ctrl_step)
        for i in range(len(interval)):
            if i+1 < len(interval):
                axs[0,0].plot(X[interval[i]:interval[i+1]+1, 0], X[interval[i]:interval[i+1]+1, 1], colors[i%6], linewidth=0.8)
                axs[0,1].plot(X[interval[i]:interval[i+1]+1, 0], X[interval[i]:interval[i+1]+1, 3], colors[i%6], linewidth=0.8)
                axs[1,0].plot(X[interval[i]:interval[i+1]+1, 1], X[interval[i]:interval[i+1]+1, 4], colors[i%6], linewidth=0.8)
                axs[1,1].plot(X[interval[i]:interval[i+1]+1, 2], X[interval[i]:interval[i+1]+1, 5], colors[i%6], linewidth=0.8)
            else:
                axs[0,0].plot(X[interval[i]:, 0], X[interval[i]:, 1], colors[i%6], linewidth=0.8)
                axs[0,1].plot(X[interval[i]:, 0], X[interval[i]:, 3], colors[i%6], linewidth=0.8)
                axs[1,0].plot(X[interval[i]:, 1], X[interval[i]:, 4], colors[i%6], linewidth=0.8)
                axs[1,1].plot(X[interval[i]:, 2], X[interval[i]:, 5], colors[i%6], linewidth=0.8)
    # axs[0,0].set_xlim([-0.1,0.2])
    # axs[0,0].set_ylim([-0.3,0.1])
    # axs[0,1].set_xlim([-0.1,0.2])
    # axs[0,1].set_ylim([-0.4,0.6])
    # axs[1,0].set_xlim([-0.3,0.1])
    # axs[1,0].set_ylim([-0.4,0.6])
    # axs[1,1].set_xlim([-0.01,0.09])
    # axs[1,1].set_ylim([-0.4,0.4])
    axs[0,0].set_xlabel('x')
    axs[0,0].set_ylabel('y')
    axs[0,1].set_xlabel('x')
    axs[0,1].set_ylabel('vx')
    axs[1,0].set_xlabel('y')
    axs[1,0].set_ylabel('vy')
    axs[1,1].set_xlabel('z')
    axs[1,1].set_ylabel('vz')
    # axs[0,0].set_title('x-y')
    # axs[0,1].set_title('x-vx')
    # axs[1,0].set_title('y-vy')
    # axs[1,1].set_title('z-vz')
    plt.show()

def simulate_one_trajectory(x0, model, ctrl_func, n_steps = 6, ctrl_step = 2e-1):
    x = x0
    # x_min = x_max = x
    # u_min = u_max = ctrl_func(x, model)
    X = np.zeros((n_steps*10, 6))
    for i in range(n_steps):
        xs, u = simulate_one_step(x, model, ctrl_func, ctrl_step, i)
        x = xs[-1]
        X[i*10:(i+1)*10] = xs
    # print("x_min =", x_min, "\nx_max =", x_max)
    # print("u_min =", u_min, "\nu_max =", u_max)
    return X

def read_gnuplot_file(filename = "polar_quadrotor_verisig_30_steps_x_y_1_0.01.plt"):
    data = []
    with open("outputs/"+filename, 'r') as f:
        tmp = []
        for line in f:
            vs = line.strip()
            if not vs or vs[0].isalpha():
                if tmp:
                    data.append(np.array(tmp))
                    tmp = []
                continue
            else:
                vs = [float(v) for v in vs.split(' ')]
                tmp.append(vs)
        if tmp:
            data.append(np.array(tmp))
            tmp = []
    return data


    

if __name__ == '__main__':
    tf_model, input_size = load_yml_model(network='tanh20x20')
    simulation(tf_model, n_trajectory=20, n_steps=30, ctrl_step=2e-1, interval=[0, 0])
