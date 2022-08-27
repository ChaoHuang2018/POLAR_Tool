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

disturb_range = 0; % Disturbance range

formatSpec = '%f %f %f\n';

fileID = fopen('nn_rl_simulation.txt','w');




for m=1:10
x1 = 90 + 20 * rand(1);
x2 = 32 + 0.2 * rand(1);
x3 = 0;
x4 = 10 + 1 * rand(1);
x5 = 30 + 0.2 * rand(1);
x6 = 0;

x = [x1;x2;x3;x4;x5;x6];

options = optimoptions('fmincon','Algorithm','sqp','Display','none');
uopt = zeros(N,4);

u_max = 0;

% Apply the control input constraints
LB = -3*ones(N,1);
UB = 3*ones(N,1);

x_now = zeros(6,1);
x_next = zeros(6,1);
z = zeros(1,1);

x_now = x;

simulation_result = x_now;

time_sequence = [];
dis = [];
for ct = 1:(Duration/Ts)
    time_sequence = [time_sequence ct*Ts];
    dis = [dis x_now(1)-x_now(4)];
    x_input = x_now;
%     x_input(1) = x_now(1);
%     x_input(2) = x_now(2);
%     x_input(3) = x_now(6);
%     x_input(4) = x_now(3) - x_now(4);
%     x_input(5) = x_now(5) - x_now(6);

    u = NN_output_rl(x_input,0,1,'controller_5_20_POLAR_mat_test');

    x_next = system_eq_dis(x_now, Ts, u);

    x = x_next;
    x_now = x_next;
end

%plot(simulation_result(15,:),simulation_result(16,:), 'blue', simulation_result(17,:),simulation_result(18,:), 'red');
plot(time_sequence, dis, 'red');

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
