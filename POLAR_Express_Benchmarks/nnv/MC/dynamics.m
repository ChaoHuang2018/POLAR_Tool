function [dx]=dynamics(x,u, T)

dx(1,1) = x(1) + x(2);
dx(2,1) = x(2) + 0.0015 * u - 0.0025 * cos(3 * x(1));
