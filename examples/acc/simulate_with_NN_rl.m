%nn_rl;
Ts = 0.1;  % Sample Time
N = 3;    % Prediction horizon
Duration = 5; % Simulation horizon

pos_radius = 10;
vel_radius = 1;
rejoin_radius = 500;
rejoin_angle = 60; 
%degree

global simulation_result;
global disturb_range;

disturb_range = 0.1; % Disturbance range

formatSpec = '%f %f %f\n';

fileID = fopen('nn_rl_simulation.txt','w');




for m=1:10
x0 = 30;
x1 = 1.4;
x2 = 90 + 1 * rand(1);
x3 = 10 + 1 * rand(1);
x4 = 32 + 0.05 * rand(1);
x5 = 30 + 0.05 * rand(1);
x6 = 0;
x7 = 0;

x = [x0;x1;x2;x3;x4;x5;x6;x7;];

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,4);

u_max = 0;

% Apply the control input constraints
LB = -3*ones(N,1);
UB = 3*ones(N,1);

x_now = zeros(8,1);
x_next = zeros(8,1);
z = zeros(1,1);

x_now = x;

simulation_result = x_now;

for ct = 1:(Duration/Ts)
    x_input = zeros(5, 1);
    x_input(1) = 30;
    x_input(2) = 1.4;
    x_input(3) = x_now(6);
    x_input(4) = x_now(3) - x_now(4);
    x_input(5) = x_now(5) - x_now(6);

    u = NN_output_rl(x_input,0,1,'acc_tanh20x20x20_mat_');

    x_next = system_eq_dis(x_now, Ts, u);

    x = x_next;
    x_now = x_next;
end

%plot(simulation_result(15,:),simulation_result(16,:), 'blue', simulation_result(17,:),simulation_result(18,:), 'red');
plot(simulation_result(5,:),simulation_result(6,:), 'red');

% title('Acc, 'FontSize', 14)
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
