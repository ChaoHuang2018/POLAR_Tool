function [dx]=dynamics(x,control_input)

g = 9.81;
M_rotor = 0.1;
M = 1;
m = M + 4 * M_rotor;
R = 0.1;
l = 0.5;
Jx = 2 /5 * R^2 + 2 * l^2 * M_rotor;
Jy = Jx;
Jz = 2 / 5 * M * R^2 + 4 * l^2 * M_rotor;

dx(1,1)= cos(x(8)) * cos(x(9)) * x(4) + x(5) * (sin(x(7)) * sin(x(8)) * cos (x(9)) - cos(x(7)) * sin(x(9))) + x(6) * (cos(x(7)) * sin(x(8)) * cos(x(9)) + sin(x(7)) * sin(x(9)));
dx(2,1)= cos(x(8)) * sin(x(9)) * x(4) + x(5) * (sin(x(7)) * sin(x(8)) * sin (x(9)) + cos(x(7)) * cos(x(9))) + x(6) * (cos(x(7)) * sin(x(8)) * sin(x(9)) - sin(x(7)) * cos(x(9)));
dx(3,1)= sin(x(8)) * x(4) - sin(x(7)) * cos(x(8)) * x(5) - cos(x(7)) * cos(x(8)) * x(6);
dx(4,1) = x(12) * x(5) - x(11) * x(6) - g * sin(x(8));
dx(5,1) = x(10) * x(6) - x(12) * x(4) + g * cos(x(8)) * sin(x(7));
dx(6,1) = x(11) * x(4) - x(10) * x(5) + g * cos(x(8)) * cos(x(7)) - g - control_input(1) / m ;
dx(7,1) = x(10) + sin(x(7)) * tan(x(8)) * x(11) + cos(x(7)) * tan(x(8)) * x(12);
dx(8,1) = cos(x(7)) * x(11) - sin(x(7)) * x(12);
dx(9,1) = sin(x(7)) / cos(x(8)) * x(11) + cos(x(7)) / cos(x(8)) * x(12);
dx(10,1) = (Jy - Jz) / Jx * x(11) * x(12) + 1 / Jx * control_input(2);
dx(11,1) = (Jz - Jx) / Jy * x(10) * x(12) + 1 / Jy * control_input(3);
dx(12,1) = (Jx - Jy) / Jz * x(10) * x(11) + 1 / Jz * 0;
dx(13,1) = 1;
