function completed = example_neuralNet_reach_09_QUAD
% example_neuralNet_reach_09_QUAD - example of reachability analysis
%                                       for an neural network controlled
%                                       system
%
% Syntax:
%    completed = example_neuralNet_reach_01_unicycle()
%
% Inputs:
%    no
%
% Outputs:
%    completed - boolean
%
% Reference:
%   [1] Johnson, Taylor T., et al. "ARCH-COMP22 Category Report:
%       Artificial Intelligence and Neural Network Control Systems (AINNCS)
%       for Continuous and Hybrid Systems Plants."
%       EPiC Series in Computing TBD (2022): TBD.
%
% Author:       Tobias Ladner
% Written:      15-June-2022
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------

disp("BENCHMARK: Quadrotor (QUAD)")

% Parameter ---------------------------------------------------------------

params.tFinal = 5;
w = 0.4;
params.R0 = polyZonotope(interval( ...
    [-w; -w; -w; -w; -w; -w; 0; 0; 0; 0; 0; 0], ...
    [w; w; w; w; w; w; 0; 0; 0; 0; 0; 0] ...
));

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.04;
options.taylorTerms = 3;
options.zonotopeOrder = 250;
% options.intermediateOrder = 200;
% options.errorOrder = 15;

options.alg = 'lin';
options.tensorOrder = 2;

% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.bound_approx = true;
evParams.polynomial_approx = "adaptive";
evParams.add_approx_error_to_Grest = true;
evParams.remove_Grest = false;
% evParams.num_generators = 10000;

% System Dynamics ---------------------------------------------------------

% open-loop system
f = @dynamics_quad;
sys = nonlinearSys(f);

% load neural network controller
% [12, 64, 64, 3]
nn = NeuralNetwork.readONNXNetwork('quad_controller_3_64_torch.onnx');
nn.evaluate(params.R0, evParams);
nn.refine(2, "layer", "both", params.R0.c, true);

% construct neural network controlled system
sys = neurNetContrSys(sys, nn, 0.1);

% Specification -----------------------------------------------------------

goalSet = interval( ...
    [-Inf;-Inf; 0.94;-Inf;-Inf;-Inf;-Inf;-Inf;-Inf;-Inf;-Inf;-Inf], ...
    [ Inf; Inf; 1.06; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf; Inf] ...
);

% Simulation --------------------------------------------------------------

tic
simRes = simulateRandom(sys, params);
tSim = toc;
disp(['Time to compute random Simulations: ', num2str(tSim)]);

% Check Violation --------------------------------------------------------

tic
isVio = false;
for i = 1:length(simRes.x)
    isVio = isVio || ~goalSet.in(simRes.x{i}(end, :)');
end
tVio = toc;
disp(['Time to check Violation in Simulations: ', num2str(tVio)]);


if isVio 
    disp("Result: VIOLATED")
    R = params.R0;
    tComp = 0;
    tVeri = 0;
else
    % Reachability Analysis -----------------------------------------------
    tic
    R = reach(sys, params, options, evParams);
    tComp = toc;
    disp(['Time to compute Reachable Set: ', num2str(tComp)]);


    % Verification --------------------------------------------------------

    tic
    Rend = R(end).timePoint.set{end};

    Rend = interval(project(Rend, 3), 'split');
    goalSet = project(goalSet, 3);

    isVeri = goalSet.in(Rend);
    tVeri = toc;
    disp(['Time to check Verification: ', num2str(tVeri)]);

    if isVeri
        disp('Result: VERIFIED');
    else
        disp('Result: UNKOWN')
    end
end

disp(['Total Time: ', num2str(tSim+tVio+tComp+tVeri)]);

% Visualization -----------------------------------------------------------
disp("Plotting..")

figure;
hold on;
box on;
gs = fill([0 0 params.tFinal, params.tFinal], [0.94, 1.06, 1.06, 0.94], 'g');
rs = plotOverTime(R, 3, 'FaceColor', [.8, .8, .8], 'EdgeColor', 'none');
is = plot(params.R0, [12, 3], 'FaceColor', 'w', 'EdgeColor', 'k');
ss = plotOverTime(simRes, 3, 'k');
xlabel('time');
ylabel('altidute');
ylim([-0.5, 1.6])
legend([gs rs, is, ss], "Goal Set", "Reachable Set", "Initial Set", "Simulations", Location="best")
set(gca,'FontSize',16)

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
%export_fig ../Benchmarks/benchmark1_sigmoid.pdf
exportgraphics(gca, 'CORA_QUAD20_soundness_issue.pdf','BackgroundColor','none')

% example completed
completed = 1;

%------------- END OF CODE --------------
