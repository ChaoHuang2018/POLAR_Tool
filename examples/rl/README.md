run `simulate_with_NN_rl.m`

* Dynamics:

    *  21 state variables x[1, 2, ..., 8]
    
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
        
        *  x(13): wingman's heading (deg) 
        
        *  x(14): lead's heading (deg)
        
        *  x(15): wingman's x-position 
        
        *  x(16): wingman's y-position
        
        *  x(17): lead's x-position x(17) = x(1) + x(15)
        
        *  x(18): lead's y-position    x(18) = x(2) + x(16)
        
        *  x(19): lead relative x axis velocity in wingman's reference x(19) = sqrt(x(20) * x(20) + x(21) * x(21))
        
        *  x(20): lead relative x axis velocity in wingman's reference x(20) = x(11) - x(8)
        
        *  x(21): lead relative velocity in wingman's reference x(21) = x(12) - x(9)
    
    *  4 control variables u[1, 2]
        
        *  u(1): lead's heading angular velocity === 0 
        
        *  u(2): lead's acceleration === 0
        
        *  u(3): wingman's heading angular velocity u(3) = deriv_t(x(13))
        
        *  u(4): wingman's acceleration u(4) = deriv_t(x(7))

*  NN controller:

    *  2 inputs:
       
        *   x_input(1) = x(1) / 1000.0
        
        *  x_input(2) = x(2)
        
        *  x_input(3) = x(3)
        
        *  x_input(4) = x(4) / 1000.0
        
        *  x_input(5) = x(5)
        
        *  x_input(6) = x(6)
        
        *  x_input(7) = x_input(8) = x_input(9) = 0???  wingman's velocity in wingman's reference???
        
        *  x_input(10) = x(19) / 400.0
        
        *  x_input(11) = x(20)
        
        *  x_input(12) = x(21)

    *  4 outputs:
        
        *  y(1): wingman's angular velocity y(1) = u(3)
        
        *  y(2): wingman's angular velocity std 
        
        *  y(3): wingman's acceleration y(3) = u(4)
        
        *  y(4): wingmna's accelration std



*  Files:
    
    *  `simulate_with_NN_rl.m` is the main simulation file
    
    *  `system_eq_dis.m` describes the dynamics model
    
    *  `NN_output_rl.m` reads NN outputs
    
