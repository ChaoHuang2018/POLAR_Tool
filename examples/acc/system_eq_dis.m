function final_val = system_eq_dis(x_initial,time, control_input)

global simulation_result;
global disturb_range;

function dxdt = tora(t,x)
    t
    u = control_input
    dxdt =[ 0;
            0;
            x(5);
            x(6);
            x(7);
            x(8);
            -2 * 2 - 2 * x(7) - 0.0001 * x(3) * x(3);
            2 * u - 2 * x(8) - 0.0001 * x(4) * x(4);
            ]

end

[t ,y] = ode45(@tora, [0 time],x_initial);

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end