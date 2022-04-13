## POLAR+Flowstar
    
    *   Run `make clean' then  './run_docking_v1.sh` to use POLAR to verify the docking network

    *   The results, including the `.m` and `.txt` files will be in './outputs/docking_v1_docking_tanh64x64'

## Alpha-beta crown + Flowstar
    
    *   Run `make flowstar_1step_v1`; go to parent dictory POLAR_Tool/examples`; run `python abcrown_flowstar_verifier.py --config ./docking/docking_v4.yaml`
    
    *   The results, including the `.m` and `.txt` files will be in `POLAR_Tool/examples/outputs/abcrown_flowstar_docking_tanh64x64_v4_crown_flowstar/`.
    
    *   The verification stops at step 14.
    
## Matlab simulation
   *    run `simulate_with_NN_rl.m` for matlab simulation

    *  Files:
    
        *  `simulate_with_NN_rl.m` is the main simulation file
    
        *  `system_eq_dis.m` describes the dynamics model
    
        *  `NN_output_rl.m` reads NN outputs

## run Verisig

    *   In the `./verisig` directory, find the `docking_tanh64x64.yml` and `docking64x64_tanh_1.model`

    *   Download and run verisig 2.0

    *   Verisig stops at step 7 and explodes severely.
