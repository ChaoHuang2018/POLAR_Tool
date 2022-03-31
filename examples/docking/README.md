## run `make clean && ./run_docking_v1.sh` to verify the docking network
    
* Neural network:
    
    * The trained network is a 64x64 tanh fully connected network.

    * The network file is `docking_tanh64x64`.

* Error log:

------Neuron 0 -------

Input remainder: [ 0.000000000000000e+00 , 0.000000000000000e+00 ]

------Neuron 1 -------

Input remainder: [ 0.000000000000000e+00 , 0.000000000000000e+00 ]

------Neuron 2 -------

Input remainder: [ 0.000000000000000e+00 , 0.000000000000000e+00 ]

------Neuron 3 -------

Input remainder: [ 0.000000000000000e+00 , 0.000000000000000e+00 ]

------------- Layer 3 starts. -------------

Output Layer 3

Output Layer 3

neural network output range by TMP: [ -6.624920681967504e-01 , -3.210755136670941e-01 ]

Neural network taylor remainder: [ -1.787342878913636e-02 , 1.787342878913636e-02 ]	

[ -6.637709832156288e-02 , 6.637709832156288e-02 ]	

[ -2.783948312080496e-02 , 2.783948312080496e-02 ]	

[ -6.836827707299650e-02 , 6.836827707299650e-02 ]	





## run `simulate_with_NN_rl.m` for matlab simulation

*  Files:
    
    *  `simulate_with_NN_rl.m` is the main simulation file
    
    *  `system_eq_dis.m` describes the dynamics model
    
    *  `NN_output_rl.m` reads NN outputs
