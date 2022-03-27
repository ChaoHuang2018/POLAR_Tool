function final_val = system_eq_dis(x_initial,time, control_input)

global simulation_result;
global disturb_range;
n = 0;
function dxdt = tora(t,x)
    n = n + 1
    t
    u = control_input
    dxdt =[ 0;
            0;
            x(5);
            x(6);
            x(7);
            x(8);
            -2 * 2 - 2 * x(7) - 0.0001 * x(5) * x(5);
            2 * u - 2 * x(8) - 0.0001 * x(6) * x(6);
            ]

end

%tspan = [0 time];
%tspan = linspace(0,time, time / 0.0001);
%[t ,y] = ode45(@tora, tspan,x_initial);

simulate_step = 1e-4;
x = zeros(8, 1);
x(:) = x_initial(:);
u = control_input;
for m = 1: (time / simulate_step)
    x_next = zeros(8, 1);
    x_next(1:2) = x(1:2);
    x_next(3) = x(3) + x(5) * simulate_step;
    x_next(4) = x(4) + x(6) * simulate_step;
    x_next(5) = x(5) + x(7) * simulate_step;
    x_next(6) = x(6) + x(8) * simulate_step;
    x_next(7) = x(7) + (-2 * 2 - 2 * x(7) - 0.0001 * x(5) * x(5)) * simulate_step;
    x_next(8) = x(8) + (2 * u - 2 * x(8) - 0.0001 * x(6) * x(6)) * simulate_step;
    x = x_next;
end

y = x';

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end