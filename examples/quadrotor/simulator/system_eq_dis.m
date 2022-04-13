function final_val = system_eq_dis(x_initial,time, control_input, step)

global simulation_result;

target_v = [-0.25, 0.25];
if step > 10
    target_v = [-0.25, -0.25];
end
if step > 20
    target_v = [0.0, 0.25];
end
if step > 25
    target_v = [0.25, -0.25];
end

ctrl_inputs = [
        -0.1, -0.1, 7.81;
        -0.1, -0.1, 11.81;
        -0.1, 0.1, 7.81;
        -0.1, 0.1, 11.81;
        0.1, -0.1, 7.81;
        0.1, -0.1, 11.81;
        0.1, 0.1, 7.81;
        0.1, 0.1, 11.81;
        ];

[~,u_id] = max(control_input);

ctrl = ctrl_inputs(u_id,:);


function dxdt = system_cont(t,x)
    
    dxdt = [  x(4) + target_v(1);
              x(5) + target_v(2);
              x(6);
              9.81*tan(ctrl(1));
              -9.81*tan(ctrl(2));
              -9.81+ctrl(3)];

end

[t ,y] = ode45(@system_cont, [0 time],x_initial);

simulation_result = [simulation_result y'];

% disp(y');

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end