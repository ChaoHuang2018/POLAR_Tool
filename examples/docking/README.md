## POLAR+Flowstar
    
    *   Run `make clean' then  './run_docking_v4.sh` to use POLAR to verify the docking network

    *   The results, including the `.m` and `.txt` files will be in './outputs/docking_v4_docking_tanh64x64_tanh'
    

## Alpha-beta crown + Flowstar
    
    *   Run `make flowstar_1step_v1`; go to parent dictory POLAR_Tool/examples`; run `python abcrown_flowstar_verifier.py --config ./docking/docking_v5.yaml`
    
    *   The results, including the `.m` and `.txt` files will be in `POLAR_Tool/examples/outputs/abcrown_flowstar_docking_tanh64x64_tanh_v5_crown_flowstar/`.
    
    *   The verification stops at step 14.
    
## Matlab simulation
   *    run `simulate_with_NN_rl.m` for matlab simulation

    *  Files:
    
        *  `simulate_with_NN_rl.m` is the main simulation file
    
        *  `system_eq_dis.m` describes the dynamics model
    
        *  `NN_output_rl.m` reads NN outputs

## run Verisig

    *   In the `./verisig` directory, find the `docking_tanh64x64.yml` and `docking64x64_tanh_3.model`

    *   Download and run verisig 2.0

    *   Verisig stops at step 7 and explodes severely.
    
    *   Find the `docking_64x64tanh_3_verisig2_x5x6_120steps.m` file in the output folder.
    
## run NNV
    
    *   In the './nnv' directory, run `run_system.m` in matlab.

    *   At step 13, NNV falsely verifies the system to be unsafe. Hence we need it to run at most 13 steps.
    
    *   The result will be in the `nnv_flowpipe.mat` file.
