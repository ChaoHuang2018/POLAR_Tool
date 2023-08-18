
read_nn_file("quad_controller_3_64");

weights = load('net.mat', 'weights').weights;
bias = load('net.mat', 'bias').bias;
activations = load('net.mat', 'activations').activations;
n = length(weights);
Layers = [];
for i=1:n
    L = LayerS(weights{i}', bias{i}', activations(i));
    Layers = [Layers L];
end
controller = FFNNS(Layers); 
% /* car model
Tr = 0.02; % reachability time step for the plant
Tc = 0.1; % control period of the plant
% output matrix
C = [1 0 0 0 0 0 0 0 0 0 0 0 0;
     0 1 0 0 0 0 0 0 0 0 0 0 0;
     0 0 1 0 0 0 0 0 0 0 0 0 0;
     0 0 0 1 0 0 0 0 0 0 0 0 0;
     0 0 0 0 1 0 0 0 0 0 0 0 0;
     0 0 0 0 0 1 0 0 0 0 0 0 0;
     0 0 0 0 0 0 1 0 0 0 0 0 0;
     0 0 0 0 0 0 0 1 0 0 0 0 0;
     0 0 0 0 0 0 0 0 1 0 0 0 0;
     0 0 0 0 0 0 0 0 0 1 0 0 0;
     0 0 0 0 0 0 0 0 0 0 1 0 0;
     0 0 0 0 0 0 0 0 0 0 0 1 0]; % output matrix
num_states = 13;
num_inputs = 3;
system = NonLinearODE(num_states, num_inputs, @dynamics, Tr, Tc, C);

% /* system
ncs = NonlinearNNCS(controller, system);
load('nnv_flowpipes.mat')
map_mat = [0 0 0 0 0 0 0 0 0 0 0 0 1;
            0 0 1 0 0 0 0 0 0 0 0 0 0];
map_vec = [];
all_flowpipes = ncs.getOutputReachSet(map_mat, map_vec);
last_flowpipe = Star.get_hypercube_hull(all_flowpipes{length(all_flowpipes)});
ncs.plotOutputReachSets('yellow', map_mat, map_vec);