function [dx]=dynamics(x,u)

dx(1,1)=x(2) - x(1)^3;
dx(2,1) = 8*u-4;
