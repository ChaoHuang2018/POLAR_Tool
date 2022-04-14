% /* An example of verifying a continuous nonlinear NNCS */
% / FFNN controller

clc
clear

weights = load('docking_tanh64x64_weights.mat');
bias = load('docking_tanh64x64_biases.mat');
n = length(fieldnames(weights));
Layers = [];
for i=1:n - 1
    cur_layer = strcat('layer', int2str(i));
    L = LayerS(weights.(cur_layer)', bias.(cur_layer)', 'tansig');
    Layers = [Layers L];
end
cur_layer = strcat('layer', int2str(n));
L = LayerS(weights.(cur_layer)', bias.(cur_layer)', 'tansig');
Layers = [Layers L];
controller = FFNNS(Layers); 
% /* car model
Tr = 0.005; % reachability time step for the plant
Tc = 1; % control period of the plant
% output matrix
C = [1/1000.0 0 0 0 0 0; 0 1/1000.0 0 0 0 0; 0 0 2 0 0 0; 0 0 0 2 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; % output matrix
num_states = 6;
num_inputs = 4;
agent = NonLinearODE(num_states, num_inputs, @dynamics, Tr, Tc, C);

% /* system
ncs = NonlinearNNCS(controller, agent); 

% /* ranges of initial set of states of the plant
lb = [24; 24; -0.13776233054248638; -0.13776233054248638; 0.1948253562373095; 0.2697150717707441]
ub = [26; 26; -0.13776233054248638; -0.13776233054248638; 0.1948253562373095; 0.27552466108497276]

% /* reachability parameters
reachPRM.init_set = Star(lb, ub);
reachPRM.ref_input = [];
reachPRM.numSteps = 13;
reachPRM.reachMethod = 'approx-star';
reachPRM.numCores = 1;

U = 0; % don't check safety for intermediate flowpipes

[safe, counterExamples, verifyTime] = ncs.verify(reachPRM, U);

%% compute reachability safety property at the end
map_mat = [0 0 0 0 1 0; 0 0 0 0 0 1];
map_vec = [];
all_flowpipes = ncs.getOutputReachSet(map_mat, map_vec);
last_flowpipe = Star.get_hypercube_hull(all_flowpipes{length(all_flowpipes)})

if last_flowpipe.ub(1) >= last_flowpipe.lb(2)
    safe = 'UNSAFE';
end

safe
verifyTime

%% save flowpipes
save('nnv_flowpipes.mat', 'ncs');
