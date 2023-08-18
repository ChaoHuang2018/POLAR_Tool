clc;
clear;

nnv_reach = load('nnv_flowpipes.mat');

fig = figure('Color', [1,1,1]);
set(fig, 'Position', [100 100 800 600])
map_mat = [1 0; 0 1];
map_vec = [];
hold on;

nnv_reach.ncs.plotOutputReachSets('blue', map_mat, map_vec);


% plot(goal_x, goal_y, 'magenta', 'linewidth', 2);
set(gca,'fontsize',24)
ylabel('x_1', 'FontSize',30);
xlabel('x_0', 'FontSize',30);
grid on;

export_fig('reach_sets_nnv_ex2_sig.pdf', '-transparent')