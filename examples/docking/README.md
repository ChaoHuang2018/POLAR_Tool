## run `make clean && ./run_docking_v1.sh` to verify the docking network
    
* Neural network:
    
    * The trained network is a 64x64 tanh fully connected network.

    * The network file is `docking_tanh64x64`.

## run `simulate_with_NN_rl.m` for matlab simulation

*  Files:
    
    *  `simulate_with_NN_rl.m` is the main simulation file
    
    *  `system_eq_dis.m` describes the dynamics model
    
    *  `NN_output_rl.m` reads NN outputs
