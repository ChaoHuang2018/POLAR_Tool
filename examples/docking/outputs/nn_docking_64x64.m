%nn_rl;
Ts = 1;  % Sample Time
N = 3;    % Prediction horizon
Duration = 120 * 1; % Simulation horizon

pos_radius = 1;
ang_radius = 360;
rejoin_radius = 500;
rejoin_angle = 60; 
%degree

global simulation_result;
global disturb_range;

disturb_range = 0.1; % Disturbance range

formatSpec = '%f %f %f\n';

fileID = fopen('../nn_rl_simulation.txt','w');




for m=1:20
x0 = 25 + 2 * rand(1) - 1; %125; % + pos_radius*rand(1);
x1 = 25 + 2 * rand(1) - 1; %125; % + pos_radius*rand(1);
%x2 = (0.2 + 2.0 * 0.001027 * sqrt(x0 * x0 + x1 * x1)) * (-0.5); % * rand(1) * cosd(ang_radius * rand(1));
%x3 = (0.2 + 2.0 * 0.001027 * sqrt(x0 * x0 + x1 * x1)) * (-0.5); % * rand(1) * sind(ang_radius * rand(1));
x2 = (0.2 + 2.0 * 0.001027 * sqrt(26 * 26 + 26 * 26)) * (-0.5); % * rand(1) * cosd(ang_radius * rand(1));
x3 = (0.2 + 2.0 * 0.001027 * sqrt(26 * 26 + 26 * 26)) * (-0.5); % * rand(1) * sind(ang_radius * rand(1));
x4 = sqrt(x2 * x2 + x3 * x3);
x5 = (0.2 + 2.0 * 0.001027 * sqrt(x0 * x0 + x1 * x1));
%x5 = (0.2 + 2.0 * 0.001027 * sqrt(26 * 26 + 26 * 26));


x = [x0;x1;x2;x3;x4;x5];

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,4);

u_max = 0;

% Apply the control input constraints
LB = -3*ones(N,4);
UB = 3*ones(N,4);

x_now = zeros(6,1);
x_next = zeros(6,1);
z = zeros(4,1);

x_now = x;

simulation_result = x_now;

for ct = 1:(Duration/Ts)
    disp(ct);
    x_input = x_now(1:6, 1);
    x_input([1, 2]) = x_input([1, 2]) / 1000.0;
    x_input([3, 4]) = x_input([3, 4]) / 0.5;

    u = NN_output_rl(x_input,0,1,'../docking_tanh64x64_mat');
    u_tmp = zeros(2,1);
    u_tmp(1) = u(1);
    u_tmp(2) = u(3);
     
    %u = zeros(4, 1);
    disp(u_tmp);
    x_next = system_eq_dis(x_now, Ts, u_tmp);

    x = x_next;
    x_now = x_next;
end

%plot(simulation_result(15,:),simulation_result(16,:), 'blue', simulation_result(17,:),simulation_result(18,:), 'red');
%plot(simulation_result(1,:),simulation_result(2,:), 'red');
plot(simulation_result(5,:), simulation_result(6,:), 'red');
% title('RL 2D Docking', 'FontSize', 14)
% xlabel('x1', 'FontSize', 14);
% ylabel('x2', 'FontSize', 14);
set(gca,'FontSize',16)
hold on;

fprintf(fileID, formatSpec, simulation_result(1:3,:));

end
fclose(fileID);

% plot( [0.1, 0.4, 0.4, 0.1, 0.1], [0.15,0.15,0.45,0.45,0.15], 'color' , [72/255 130/255 197/255], 'LineWidth', 2.0);
% hold on;
% 
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% print(fig,'../rl/simulation','-dpdf')
% export_fig ../rl/simulation.pdf
