function final_val = system_eq_dis(x_initial,time, control_input)

global simulation_result;
global disturb_range;

function dxdt = tora(t,x)
 
    u = control_input;
    dxdt =[ x(3);
            x(4);
            2.0 * 0.001027 * x(4) + 3 * 0.001027 * 0.001027 * x(1) + u(1) / 12.;
            -2.0 * 0.001027 * x(3) + u(2) / 12.;
            ((2.0 * 0.001027 * x(4) + 3 * 0.001027 * 0.001027 * x(1) + u(1) / 12.) * x(3) + (-2.0 * 0.001027 * x(3) + u(2) / 12.) * x(4)) / x(5);
            2.0 * 0.001027 * (x(1) * x(3) + x(2) * x(4)) / sqrt(x(1) * x(1) + x(2) * x(2));
            ];

end

[t ,y] = ode45(@tora, [0 time],x_initial);

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end