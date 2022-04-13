# Quadrotor example
This is the Quadrotor example that also used in [Verisig](https://https://github.com/Verisig/verisig/tree/master/examples/quadrotor/).

## Experiment results:
We verify the quadrotor controller for 30 control steps.

### Simulation result
We compare our result with several simulated trajectories
![Simulation and Pola flowpipes](simulation.png)

### Baseline
We compare our results with the Verisig result. Verisig verification takes 1385 seconds.
![Verisig x-y flowpipes](outputs/autosig.png)

### Polar with symbolic remainder

#### Flowstar stepsize = 0.01
Polar Verification takes 186 seconds.
![Polar w/ SR, stepsize = 0.01, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_1_0.01.png)
![Polar w/ SR, stepsize = 0.01, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_1_0.01.png)
![Polar w/ SR, stepsize = 0.01, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_1_0.01.png)
![Polar w/ SR, stepsize = 0.01, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_1_0.01.png)

#### Flowstar stepsize = 0.02
Polar Verification takes 108 seconds.
![Polar w/ SR, stepsize = 0.02, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_1_0.02.png)
![Polar w/ SR, stepsize = 0.02, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_1_0.02.png)
![Polar w/ SR, stepsize = 0.02, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_1_0.02.png)
![Polar w/ SR, stepsize = 0.02, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_1_0.02.png)

#### Flowstar stepsize = 0.05
Polar Verification takes 63 seconds.
![Polar w/ SR, stepsize = 0.05, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_1_0.05.png)
![Polar w/ SR, stepsize = 0.05, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_1_0.05.png)
![Polar w/ SR, stepsize = 0.05, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_1_0.05.png)
![Polar w/ SR, stepsize = 0.05, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_1_0.05.png)

#### Flowstar stepsize = 0.1
Polar Verification takes 57 seconds.
![Polar w/ SR, stepsize = 0.1, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_1_0.10.png)
![Polar w/ SR, stepsize = 0.1, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_1_0.10.png)
![Polar w/ SR, stepsize = 0.1, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_1_0.10.png)
![Polar w/ SR, stepsize = 0.1, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_1_0.10.png)


### Polar without symbolic remainder

#### Flowstar stepsize = 0.01
Polar Verification takes 182 seconds.
![Polar w/o SR, stepsize = 0.01, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_0_0.01.png)
![Polar w/o SR, stepsize = 0.01, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_0_0.01.png)
![Polar w/o SR, stepsize = 0.01, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_0_0.01.png)
![Polar w/o SR, stepsize = 0.01, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_0_0.01.png)

#### Flowstar stepsize = 0.02
Polar Verification takes 110 seconds.
![Polar w/o SR, stepsize = 0.02, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_0_0.02.png)
![Polar w/o SR, stepsize = 0.02, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_0_0.02.png)
![Polar w/o SR, stepsize = 0.02, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_0_0.02.png)
![Polar w/o SR, stepsize = 0.02, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_0_0.02.png)

#### Flowstar stepsize = 0.05
Polar Verification takes 63 seconds.
![Polar w/o SR, stepsize = 0.05, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_0_0.05.png)
![Polar w/o SR, stepsize = 0.05, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_0_0.05.png)
![Polar w/o SR, stepsize = 0.05, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_0_0.05.png)
![Polar w/o SR, stepsize = 0.05, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_0_0.05.png)

#### Flowstar stepsize = 0.1
Polar Verification takes 56 seconds.
![Polar w/o SR, stepsize = 0.1, x-y flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_y_0_0.10.png)
![Polar w/o SR, stepsize = 0.1, x-vx flowpipes](outputs/polar_quadrotor_verisig_30_steps_x_vx_0_0.10.png)
![Polar w/o SR, stepsize = 0.1, y-vy flowpipes](outputs/polar_quadrotor_verisig_30_steps_y_vy_0_0.10.png)
![Polar w/o SR, stepsize = 0.1, z-vz flowpipes](outputs/polar_quadrotor_verisig_30_steps_z_vz_0_0.10.png)