% /* An example of verifying a continuous nonlinear NNCS */
% / FFNN controller

clc
clear

weights = load('weights.mat');
bias = load('biases.mat');
n = length(fieldnames(weights));
Layers = [];
for i=1:n - 1
    cur_layer = strcat('layer', int2str(i));
    L = LayerS(weights.(cur_layer)', bias.(cur_layer)', 'tansig');
    Layers = [Layers L];
end
cur_layer = strcat('layer', int2str(n));
L = LayerS(weights.(cur_layer)', bias.(cur_layer)', 'purelin');
Layers = [Layers L];
controller = FFNNS(Layers); 
% /* car model
Tr = 0.001; % reachability time step for the plant
Tc = 0.1; % control period of the plant
% output matrix
C = [0 0 0 0 1 0;1 0 0 -1 0 0; 0 1 0 0 -1 0]; % output matrix
num_states = 6;
num_inputs = 1;
car = NonLinearODE(num_states, num_inputs, @car_dynamics, Tr, Tc, C);

% /* system
ncs = NonlinearNNCS(controller, car); 

% /* ranges of initial set of states of the plant
lb = [90; 32; 0; 10; 30; 0];
ub = [91; 32.05; 0; 11; 30.05; 0];

% /* reachability parameters
reachPRM.init_set = Star(lb, ub);
reachPRM.ref_input = [30; 1.4];
reachPRM.numSteps = 50;
reachPRM.reachMethod = 'approx-star';
reachPRM.numCores = 1;

U = 0; % don't check safety for intermediate flowpipes

[safe, counterExamples, verifyTime] = ncs.verify(reachPRM, U);

%% compute reachability safety property at the end
map_mat = [0 1 0 0 0 0; 0 0 0 0 1 0];
map_vec = [];
all_flowpipes = ncs.getOutputReachSet(map_mat, map_vec);
last_flowpipe = Star.get_hypercube_hull(all_flowpipes{length(all_flowpipes)});

x2_ub = 22.87;
x2_lb = 22.81;
x5_ub = 30.02;
x5_lb = 29.88;

if last_flowpipe.lb(1) >= x2_ub || last_flowpipe.lb(2) >= x5_ub || last_flowpipe.ub(1) <= x2_lb || last_flowpipe.ub(2) <= x5_lb
    safe = 'UNSAFE';
elseif last_flowpipe.ub(1) >= x2_ub || last_flowpipe.ub(2) >= x5_ub || last_flowpipe.lb(1) <= x2_lb || last_flowpipe.lb(2) <= x5_lb
    safe = 'UNKNOWN';
end

safe
verifyTime

%% save flowpipes
save('nnv_flowpipes.mat', 'ncs');