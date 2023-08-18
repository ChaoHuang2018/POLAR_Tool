function [dx]=car_dynamics(x,a_ego)

mu=0.0001; % friction parameter

% lead car dynamics
a_lead = -2; 
dx(1,1)=x(2);
dx(2,1) = x(3);
dx(3,1) = -2 * x(3) + 2 * a_lead - mu*x(2)^2;
% ego car dyanmics
dx(4,1)= x(5); 
dx(5,1) = x(6);
dx(6,1) = -2 * x(6) + 2 * a_ego - mu*x(5)^2;
