function [dx]=dynamics(x,u)

dx(1,1)=-x(1)*(0.1 + (x(1) + x(2))^2);
dx(2,1) = (2*u+x(1))*(0.1 + (x(1) + x(2))^2);
