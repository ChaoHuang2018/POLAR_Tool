#!/usr/bin/python
import sys
import yaml
import pickle
import numpy as np
import os

def main(argv): 
    input_filename = argv[0]
    print(input_filename)


    output_filename = input_filename.split('.pickle')[0] + ".yml"


    f = open(input_filename, 'rb')
    conf = pickle.load(f)
    wbs = conf['default_policy']
    f.close()
    
    f = open(output_filename, 'w')
    model = {'activations': [], 'offsets': {}, 'weights': {}}

    model['activations'] = {1: 'Tanh', 2: 'Tanh', 3: 'Linear'}
     
    model['offsets'][1] = []
    model['weights'][1] = []
    w = wbs['default_policy/fc_1/kernel']
    b = wbs['default_policy/fc_1/bias']
    for i in range(b.shape[0]):
        model['weights'][1].append([])
        for j in range(w.shape[0]):
            model['weights'][1][-1].append(float(w[j][i]))
        model['offsets'][1].append(float(b[i]))
    
    model['offsets'][2] = []
    model['weights'][2] = []
    w = wbs['default_policy/fc_2/kernel']
    b = wbs['default_policy/fc_2/bias']
    for i in range(b.shape[0]):
        model['weights'][2].append([])
        for j in range(w.shape[0]):
            model['weights'][2][-1].append(float(w[j][i]))
        model['offsets'][2].append(float(b[i]))
    
    model['offsets'][3] = []
    model['weights'][3] = []
    w = wbs['default_policy/fc_out/kernel']
    b = wbs['default_policy/fc_out/bias']
    for i in range(b.shape[0]):
        model['weights'][3].append([])
        for j in range(w.shape[0]):
            model['weights'][3][-1].append(float(w[j][i]))
        model['offsets'][3].append(float(b[i]))
    
    documents = yaml.dump(model, f)
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
