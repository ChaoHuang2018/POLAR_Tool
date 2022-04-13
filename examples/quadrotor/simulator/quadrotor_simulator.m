% plot( [-0.05, 0.05, 0.05, -0.05, -0.05] , [-0.05,-0.05,0.2,0.2,-0.05] , 'color' , [72/255 130/255 197/255], 'LineWidth', 2.0);
% hold on;
clear;
% for step = 0:17
%     filename = "./outputs/attitude_control_CLF_controller_layer_num_3_crown_flowstar/x0_x3_" + string(step) + ".m";
%     run(filename);
% end
% plot_reach_sets_sig;
% attitude_control_sig_tmp_x0_x1;
% % nn_ac_sigmoid_x0_x1_0;
% nn_ac_sigmoid_x0_x1_1;
% % nn_ac_sigmoid_x4_x5_1;

Ts = 0.2;  % Sample Time
steps = 30; % Simulation horizon
number = 20;


% Things specfic to the model starts here 

% Limits of initial [-1,1]^2

global simulation_result;
global disturb_range;

disturb_range = 0; % Disturbance range

formatSpec = '%f %f\n';

fileID = fopen('nn_1_sigmoid.txt','w');

% figure;
for m=1:number
% Setting the initial state
disp(num2str(m) + "-th simulation starts: ")

x1 = -0.05 + 0.025*rand(1);
x2 = -0.025 + 0.025*rand(1);
x3 = 0;
x4 = 0;
x5 = 0;
x6 = 0;

x = [x1;x2;x3;x4;x5;x6];
pre_process = [0.2;0.2;0.2;0.1;0.1;0.1];

simulation_result = x;

% Things specfic to the model  ends here

% Apply the control input constraints

x_next = zeros(6,1);

x_now = x;

% Start simulation
for step = 1:steps
      
      u2 = NN_output(x_now .* pre_process,'tanh20x20');
     %disp(u2);
      
      x_next = system_eq_dis(x_now, Ts, u2, step);

      x = x_next;
      x_now = x_next;
end


% figure;
plot(simulation_result(2,:),simulation_result(3,:), 'color', [210/255, 95/255, 95/255]);

title('Quadrotor', 'FontSize', 14)
xlabel('${x}$','interpreter','latex', 'FontWeight','bold')
ylabel('${y}$','interpreter','latex', 'FontWeight','bold')
set(gca,'FontSize',16)
hold on;

end

% % fclose(fileID);
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% print(fig,'../Benchmarks/attitude_control_benchmark_sigmoid_x0_x1','-dpdf')
% export_fig ../Benchmarks/attitude_control_benchmark_sigmoid_x0_x1.pdf