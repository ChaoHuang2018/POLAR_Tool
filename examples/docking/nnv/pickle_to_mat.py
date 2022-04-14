#!/usr/bin/python
import sys
import yaml
import pickle
import numpy as np
import os
from scipy.io import savemat

def main(argv): 
    input_filename = argv[0]
    print(input_filename)

    
    f = open(input_filename, 'rb')
    conf = pickle.load(f)
    wbs = conf['default_policy']
    f.close()
    
    w_dict = {"layer1": wbs['default_policy/fc_1/kernel'], "layer2": wbs['default_policy/fc_2/kernel'], "layer3": wbs['default_policy/fc_out/kernel']}
    savemat(input_filename.split('.pickle')[0] + "_weights.mat", w_dict)

    b_dict = {"layer1": wbs['default_policy/fc_1/bias'], "layer2": wbs['default_policy/fc_2/bias'], "layer3": wbs['default_policy/fc_out/bias']}
    savemat(input_filename.split('.pickle')[0] + "_biases.mat", b_dict)

    act_dict = {"layer1": 'tansig', "layer2": 'tansig', "layer3": 'tansig'}
    savemat(input_filename.split('.pickle')[0] + "_activations.mat", act_dict)
  


if __name__ == "__main__":
    main(sys.argv[1:])
