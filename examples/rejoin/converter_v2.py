import pickle
import numpy as np
import os

pathname = "rejoin_tanh64x64"
f = open(os.path.join("./", pathname + ".pickle"), 'rb')
conf = pickle.load(f)
wbs = conf['default_policy']

f.close()

f = open(os.path.join("./", pathname + "_v2"), 'w')

num_inputs = wbs['default_policy/fc_1/kernel'].shape[0]
f.write("{}".format(num_inputs) + os.linesep)
num_outputs = wbs['default_policy/fc_out/bias'].shape[0]
f.write(str(num_outputs) + os.linesep)

num_of_hidden_layers = 2
num_of_hidden_layers += 4
num_of_hidden_layers += 4
f.write(str(num_of_hidden_layers) + os.linesep)

f.write(str(4 * num_inputs) + os.linesep)
f.write(str(num_inputs) + os.linesep)
f.write(str(4 * num_inputs) + os.linesep)
f.write(str(num_inputs) + os.linesep)
f.write(str(wbs['default_policy/fc_1/bias'].shape[0]) + os.linesep) 
f.write(str(wbs['default_policy/fc_2/bias'].shape[0]) + os.linesep) 
f.write(str(num_outputs) + os.linesep)
f.write(str(4 + 1 + 4 + 1) + os.linesep)
f.write(str(num_outputs) + os.linesep)
f.write(str(4 + 1 + 4 + 1) + os.linesep)


activs = ['tanh', 'tanh', 'Affine']
activs = ['ReLU', 'Affine', 'ReLU', 'Affine'] + activs
activs = activs + ['ReLU', 'Affine', 'ReLU', 'Affine']
for activ in activs:
    f.write(str(activ) + os.linesep)
 
# max(x, - 1)
w = np.zeros([num_inputs, 4 * num_inputs])
b = np.zeros([4 * num_inputs])
for i in range(num_inputs):
    for j in range(4):
        if j % 4 == 0 or (j % 4 == 3):
            w[i][j + 4 * i] = 1.
        else:
            w[i][j + 4 * i] = -1
        b[j + 4 * i] = 2. * (j % 2) - 1. 
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
w = np.zeros([4 * num_inputs, num_inputs])
b = np.zeros([num_inputs])
for i in range(num_inputs):
    for j in range(4):
        if j % 4 != 1:
            w[j + 4 * i][i] = 0.5
        else:
            w[j + 4 * i][i] = -0.5
    b[i] = 0.
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)

# min(x, 1)= - max(-x, -1)
w = np.zeros([num_inputs, 4 * num_inputs])
b = np.zeros([4 * num_inputs])
for i in range(num_inputs):
    for j in range(4):
        if j % 4 == 0 or (j % 4 == 3):
            w[i][j + 4 * i] = -1.
        else:
            w[i][j + 4 * i] = 1
        b[j + 4 * i] = 2. * (j % 2) - 1. 
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
w = np.zeros([4 * num_inputs, num_inputs])
b = np.zeros([num_inputs])
for i in range(num_inputs):
    for j in range(4):
        if j % 4 != 1:
            w[j + 4 * i][i] = -0.5
        else:
            w[j + 4 * i][i] = 0.5
    b[i] = 0.
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)

"""
# -1
w = -1 * np.eye(num_inputs)
b = np.zeros([num_inputs])
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
"""
 

w = wbs['default_policy/fc_1/kernel']
b = wbs['default_policy/fc_1/bias']
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)

w = wbs['default_policy/fc_2/kernel']
b = wbs['default_policy/fc_2/bias']
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)


w = wbs['default_policy/fc_out/kernel']
b = wbs['default_policy/fc_out/bias']
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)

# max(x[0], -0.17)
# max(x[2], -96.5)
w = np.zeros([num_outputs, 4 + 1 + 4 + 1])
b = np.zeros([4 + 1 + 4 + 1])
w[0][0] = 1.
w[0][1] = -1.
w[0][2] = -1.
w[0][3] = 1.
w[1][4] = 1.
w[2][5] = 1.
w[2][6] = -1.
w[2][7] = -1.
w[2][8] = 1.
w[3][9] = 1.

b[0] = -0.17
b[1] = 0.17
b[2] = -0.17
b[3] = 0.17
b[5] = -96.5
b[6] = 96.5
b[7] = -96.5
b[8] = 96.5
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
w = np.zeros([4 + 1 + 4 + 1, num_outputs])
b = np.zeros([num_outputs])
w[0][0] = 0.5
w[1][0] = -0.5
w[2][0] = 0.5
w[3][0] = 0.5
w[4][1] = 1.
w[5][2] = 0.5
w[6][2] = -0.5
w[7][2] = 0.5
w[8][2] = 0.5
w[9][3] = 1.
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)

# min(x[0], 1)= - max(-x[0], -0.17)
# min(x[2], 96.5) = - max(-x[2], -96.5)
w = np.zeros([num_outputs, 4 + 1 + 4 + 1])
b = np.zeros([4 + 1 + 4 + 1])
w[0][0] = -1.
w[0][1] = 1.
w[0][2] = 1.
w[0][3] = -1.
w[1][4] = 1.
w[2][5] = -1.
w[2][6] = 1.
w[2][7] = 1.
w[2][8] = -1.
w[3][9] = 1.
b[0] = -0.17
b[1] = 0.17
b[2] = -0.17
b[3] = 0.17
b[5] = -96.5
b[6] = 96.5
b[7] = -96.5
b[8] = 96.5
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
w = np.zeros([4 + 1 + 4 + 1, num_outputs])
b = np.zeros([num_outputs])
w[0][0] = -0.5
w[1][0] = 0.5
w[2][0] = -0.5
w[3][0] = -0.5
w[4][1] = 1.
w[5][2] = -0.5
w[6][2] = 0.5
w[7][2] = -0.5
w[8][2] = -0.5
w[9][3] = 1.
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)


f.write("0.0" + os.linesep)
f.write("1.0")

f.close()