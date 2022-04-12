## run `make clean && ./run_rejoin_v2.sh` to verify the rejoin network
    
* Neural network:
    
    * The trained network is a 64x64 tanh fully connected network.

    * Added a clip(-1, 1) structure before the original network structure
    
    * Two relu-affine structures are used to construct clip(-1, 1) = -1 * max(-max(x, -1), -1)
    
    * The first relu-affine structure is for max(x, -1) = ReLU(x*[1, -1, -1, 1] + [-1, 1, -1, 1]) * [0.5, -0.5, 0.5, 0.5]
    
    * The second relu-affine structure is for max(-x, -1) = ReLU(x[-1, 1, 1, -1] + [-1, 1, -1, 1]) * [0.5, -0.5, 0.5, 0.5]

    * Add a clip(-0.17, 0.17)(x(1)) and a clip(-96.5, 96.5)(x(3)) structure to the end of the original network. 

    * The network ends up with 10 hidden layers

    * The network file is `rejoin_tanh64x64_v2`.

* Dynamics are explained below. 

* In the beginning of each iteration (simulation step), a TaylorModelVec<Real> variable `tmv_temp` is used to store the current state variables in the `initial_set`

* Then a new TaylorModelVec<Real> variable  `wingman_frame_rot_mat`  is created to store the `wingman_frame_rot_mat` matrix based on the x(13), i.e., tmv_temp.tms[12]

* After having `wingman_frame_rot_mat`, the TaylorModel<Real> variables in the `tmv_temp` are transformed and stored into a new TaylorModelVec<Real> variable `tmv_input`

* `tmv_input` will be used as input to the neural network, which outputs a `tmv_output` TaylorModelVec<Real> varaible

* `tmv_output` will be stored in `initial_set` which will be fed to the dynamics model and generate the flowpipe

 
## run `simulate_with_NN_rl.m` for matlab simulation

* Dynamics:

    *  14 state variables x[1, 2, ..., 14]
    
        *  x(1): lead to wingman distance x(1) = sqrt(x(2) * x(2) + x(3) * x(3))
        
        *  x(2): lead relative x position in wingman's reference   x(2) = x(1) * cosd(x(14))
        
        *  x(3): lead relative y position in wingman's reference   x(3) = x(1) * sind(x(14))
        
        *  x(4): rejoin position to wingman distance   x(4) = sqrt(x(5) * x(5) + x(6) * x(6))
        
        *  x(5): rejoin relative x position in wingman's reference x(5) = x(1) + 500 * cosd(60 + 180 + x(14))
        
        *  x(6): rejoin relative y position in wingman's reference x(6) = x(2) + 500 * cosd(60 + 180 + x(14))
        
        *  x(7): wingman's velocity    x(7) = sqrt(x(8) * x(8) + x(9) * x(9))
        
        *  x(8): wingman's x-axis velocity x(8) = x(7) * cosd(x(13))
        
        *  x(9): wingman's y-axis velocity x(9) = x(7) * cosd(x(13))
        
        *  x(10): lead's velocity x(10) = sqrt(x(11) * x(11) + x(12) * x(12))
        
        *  x(11): lead's x-axis velocity   x(11) = x(10) * cosd(x(14))
        
        *  x(12): lead's y-axis velocity   x(12) = x(10) * cosd(x(14))
        
        *  x(13): wingman's heading  
        
        *  x(14): lead's heading 


    *  4 control variables u[1, 2]
        
        *  u(1): lead's heading angular velocity === 0 
        
        *  u(2): lead's acceleration === 0
        
        *  u(3): wingman's heading angular velocity u(3) = deriv_t(x(13))
        
        *  u(4): wingman's acceleration u(4) = deriv_t(x(7))

*  NN controller:

    *   Need a transformation matrix 

        *   wingman_frame_rot_mat = [
            cos(-wingman_heading) -sin(-wingman_heading);
            sin(-wingman_heading) cos(-wingman_heading);
        ];

    *  12 inputs:
       
        *   x_input(2:3) = wingman_frame_rot_mat * x_input(2:3);
        
        *   x_input(5:6) = wingman_frame_rot_mat * x_input(5:6);
    
        *   x_input(8:9) = wingman_frame_rot_mat * x_input(8:9);
    
        *   x_input(11:12) = wingman_frame_rot_mat * x_input(11:12);
   
        *   % normalize position vectors
    
        *   x_input(2:3) = x_input(2:3) / x_input(1);
    
        *   x_input(5:6) = x_input(5:6) / x_input(4);
   
        *   % normalize distance magnitudes

        *   x_input([1, 4]) = x_input([1, 4]) / 1000.0;
 
        *   % wingman's velocity in wingman's reference????
        
        *   %   umberto: note that by "reference frame" we mean the the wingman's
        
        *   %       local coordinates, not the inertial reference from. Velocities are
        
        *   %       not relative, only their direction is modified by the reference
        
        *   %       frame transformation
   
        *   % normalize the x,y components of the velocity by the vector magnitude
        
        *   % for the magnorm transformation
        
        *   x_input(8:9) = x_input(8:9)/x_input(7);
        
        *   x_input(11:12) = x_input(11:12) / x_input(10);
  
        *   % normalize velocities magnitudes
    
        *   x_input([7, 10]) = x_input([7, 10]) / 400.0;
    
    *  4 outputs:
        
        *  y(1): wingman's angular velocity y(1) = u(3)
        
        *  y(2): wingman's angular velocity std 
        
        *  y(3): wingman's acceleration y(3) = u(4)
        
        *  y(4): wingmna's accelration std



*  Files:
    
    *  `simulate_with_NN_rl.m` is the main simulation file
    
    *  `system_eq_dis.m` describes the dynamics model
    
    *  `NN_output_rl.m` reads NN outputs
