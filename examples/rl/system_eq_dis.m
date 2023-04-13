function final_val = system_eq_dis(x_initial,time, control_input)

global simulation_result;
global disturb_range;

function dxdt = tora(t,x)
    %%%
    rejoin_radius = 500;
    rejoin_angle = 60; 
    u = control_input;
    dxdt =[ (x(2) * (x(11) - x(8)) + x(3) * (x(12) - x(9))) / x(1);
            x(11) - x(8);
            x(12) - x(9);
            (x(5) * (x(11) - x(8) - rejoin_radius * sind(rejoin_angle + 180 + x(14)) * control_input(1)) + x(6) * (x(12) - x(9) + rejoin_radius * cosd(rejoin_angle + 180 + x(14)) * control_input(1)))/x(4);
            x(11) - x(8) - rejoin_radius * sind(rejoin_angle + 180 + x(14)) * control_input(1);
            x(12) - x(9) + rejoin_radius * cosd(rejoin_angle + 180 + x(14)) * control_input(1);
            control_input(4);
            control_input(4) * cosd(x(13)) - control_input(3) * x(7) * sind(x(13));
            control_input(4) * sind(x(13)) + control_input(3) * x(7) * cosd(x(13));
            control_input(2);
            control_input(2) * cosd(x(14)) - control_input(1) * x(10) * sind(x(14));
            control_input(2) * sind(x(14)) + control_input(1) * x(10) * cosd(x(14));
            control_input(3) * 180 / pi;
            control_input(1) * 180 / pi;
            ];
            %x(8);
            %x(9);
            %x(11);
            %x(12);
            %control_input(2) * cosd(x(14)) - control_input(1) * x(10) * sind(x(14)) - (control_input(4) * cosd(x(13)) - control_input(3) * x(7) * sind(x(13)));
            %control_input(2) * sind(x(14)) + control_input(1) * x(10) * cosd(x(14)) - (control_input(4) * sind(x(13)) + control_input(3) * x(7) * cosd(x(13)));
            %(x(19) * (control_input(2) * cosd(x(14)) - control_input(1) * x(10) * sind(x(14)) - (control_input(4) * cosd(x(13)) - control_input(3) * x(7) * sind(x(13)))) + x(20) * (control_input(2) * sind(x(14)) + control_input(1) * x(10) * cosd(x(14)) - (control_input(4) * sind(x(13)) + control_input(3) * x(7) * cosd(x(13)))))/x(21);
            %];

end

[t ,y] = ode45(@tora, [0 time],x_initial);

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end