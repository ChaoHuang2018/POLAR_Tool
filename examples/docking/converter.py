import pickle
import numpy as np
import os

pathname = "docking_tanh64x64_v1"
f = open(os.path.join("./", pathname + ".pickle"), 'rb')
conf = pickle.load(f)
wbs = conf['default_policy']

f.close()

f = open(os.path.join("./", pathname), 'w')

num_inputs = wbs['default_policy/fc_1/kernel'].shape[0]
f.write("{}".format(num_inputs) + os.linesep)
num_outputs = wbs['default_policy/fc_out/bias'].shape[0]
f.write(str(num_outputs) + os.linesep)

num_of_hidden_layers = 2
 
f.write(str(num_of_hidden_layers) + os.linesep)

f.write(str(wbs['default_policy/fc_1/bias'].shape[0]) + os.linesep) 
f.write(str(wbs['default_policy/fc_2/bias'].shape[0]) + os.linesep) 

activs = ['tanh', 'tanh', 'Affine']
 
for activ in activs:
    f.write(str(activ) + os.linesep)


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