function final_val = system_eq_dis(x_initial,time, control_input)

global simulation_result;
global rejoin_radius;
global rejoin_angle;
function dxdt = tora(t,x)
    %%%
   
    u = control_input;
    dxdt =[ (x(2) * (x(11) - x(8)) + x(3) * (x(12) - x(9))) / x(1);
            
            %(x(10) * x(11) - x(7) * x(8));
            x(11) - x(8);
            
            %(x(10) * x(12) - x(7) * x(9));
            x(12) - x(9);
            
            (x(5) * (x(11) - x(8)) + x(6) * (x(12) - x(9))) / x(4); %(x(5) * (x(11) - x(8) - rejoin_radius * sin(rejoin_angle + pi + x(14)) * control_input(1)) + x(6) * (x(12) - x(9) + rejoin_radius * cos(rejoin_angle + pi + x(14)) * control_input(1)))/x(4);
            
            
            %x(10) * x(11) - x(7) * x(8) - rejoin_radius * sin(rejoin_angle + pi + x(14)) * control_input(1);
            x(11) - x(8); %x(11) - x(8) - rejoin_radius * sin(rejoin_angle + pi + x(14)) * control_input(1);
            
            %x(10) * x(12) - x(7) * x(9) + rejoin_radius * cos(rejoin_angle + pi + x(14)) * control_input(1);
            x(12) - x(9); %x(12) - x(9) + rejoin_radius * cos(rejoin_angle + pi + x(14)) * control_input(1);
            
            control_input(4);
            %(x(8) * (control_input(4) * cos(x(13)) - x(7) * control_input(3) * sin(x(13))) + x(9) * (control_input(4) * sin(x(13)) - x(7) * control_input(3) * cos(x(13)))) / x(7);
            control_input(4) * cos(x(13)) - x(7) * control_input(3) * sin(x(13));
            control_input(4) * sin(x(13)) + x(7) * control_input(3) * cos(x(13));
            %(x(11) * (control_input(2) * cos(x(14)) - control_input(1) * sin(x(14))) + x(12) * (control_input(2) * sin(x(14)) + control_input(1) * cos(x(14)))) / x(10);
            0.0; %control_input(2);
            0.0; %control_input(2) * cos(x(14)) - control_input(1) * sin(x(14));
            0.0; %control_input(2) * sin(x(14)) + control_input(1) * cos(x(14));
            control_input(3);
            0.0; %control_input(1);
            ];
            %x(8);
            %x(9);
            %x(11);
            %x(12);
            %control_input(2) * cos(x(14)) - control_input(1) * x(10) * sin(x(14)) - (control_input(4) * cos(x(13)) - control_input(3) * x(7) * sin(x(13)));
            %control_input(2) * sin(x(14)) + control_input(1) * x(10) * cos(x(14)) - (control_input(4) * sin(x(13)) + control_input(3) * x(7) * cos(x(13)));
            %(x(19) * (control_input(2) * cos(x(14)) - control_input(1) * x(10) * sin(x(14)) - (control_input(4) * cos(x(13)) - control_input(3) * x(7) * sin(x(13)))) + x(20) * (control_input(2) * sin(x(14)) + control_input(1) * x(10) * cos(x(14)) - (control_input(4) * sin(x(13)) + control_input(3) * x(7) * cos(x(13)))))/x(21);
            %];
    
end
%[t ,y] = ode45(@tora, [0 time],x_initial);

step_length = 0.0001;
x = x_initial(:);
for t=1: (time / step_length)
    x(1) = x(1) + ((x(2) * (x(11) - x(8)) + x(3) * (x(12) - x(9))) / x(1)) * step_length;
    x(2) = x(2) + (x(11) - x(8)) * step_length;
    x(3) = x(3) + (x(12) - x(9)) * step_length;
    x(4) = x(4) + ((x(5) * (x(11) - x(8)) + x(6) * (x(12) - x(9))) / x(4)) * step_length;
    x(5) = x(5) + (x(11) - x(8)) * step_length;
    x(6) = x(6) + (x(12) - x(9)) * step_length;
    x(7) = x(7) + control_input(4) * step_length;
    x(8) = x(8) + (control_input(4) * cos(x(13)) - x(7) * control_input(3) * sin(x(13))) * step_length;
    x(9) = x(9) + (control_input(4) * sin(x(13)) + x(7) * control_input(3) * cos(x(13))) * step_length;
    x(13) = x(13) + control_input(3) * step_length;
    x(15) = x(15) + x(8) * step_length;
    x(16) = x(16) + x(9) * step_length;
    x(17) = x(17) + x(11) * step_length;
    x(18) = x(18) + x(12) * step_length;

end
    
y = x';

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end