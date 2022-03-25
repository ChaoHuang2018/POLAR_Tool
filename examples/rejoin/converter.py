import pickle
import numpy as np
import os

f = open("./weights.pickle", 'rb')
conf = pickle.load(f)
wbs = conf['default_policy']

f.close()

f = open("./rl_tanh256x256_mat", 'w')

num_inputs = wbs['default_policy/fc_1/kernel'].shape[0]
f.write("{}".format(num_inputs) + os.linesep)
num_outputs = wbs['default_policy/fc_out/bias'].shape[0]
f.write(str(num_outputs) + os.linesep)

num_of_hidden_layers = 3
num_of_hidden_layers += 5
f.write(str(num_of_hidden_layers) + os.linesep)

f.write(str(4 * num_inputs) + os.linesep)
f.write(str(num_inputs) + os.linesep)
f.write(str(4 * num_inputs) + os.linesep)
f.write(str(num_inputs) + os.linesep)
f.write(str(num_inputs) + os.linesep)
f.write(str(wbs['default_policy/fc_1/bias'].shape[0]) + os.linesep) 
f.write(str(wbs['default_policy/fc_2/bias'].shape[0]) + os.linesep) 

activs = ['tanh', 'tanh', 'Affine']
activs = ['ReLU', 'Affine', 'ReLU', 'Affine', 'Affine'] + activs
#for activ in activs:
#    f.write(str(activ) + os.linesep)

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


# -1
w = -1 * np.eye(num_inputs)
b = np.zeros([num_inputs])
for i in range(b.shape[0]):
    for j in range(w.shape[0]):
        f.write(str(w[j][i]) + os.linesep)
    f.write(str(b[i]) + os.linesep)
 
 

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


f.write("0.0" + os.linesep)
f.write("1.0")

f.close()