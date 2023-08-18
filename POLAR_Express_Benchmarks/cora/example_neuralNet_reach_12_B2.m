function completed = example_neuralNet_reach_12_B2
% example_neuralNet_reach_12_Benchmark 2 - example of reachability analysis for a
%                                   neural network controlled numerical
%                                   benchmark 2 in POLAR
%
%
% Syntax:
%    completed = example_neuralNet_reach_12_B2()
%
% Inputs:
%    no
%
% Outputs:
%    completed - boolean
%
%
% Author:       Yixuan Wang
% Written:      30-November-2022
% Last revision:---
% Note:         Adapted from the Cora repo examples

%------------- BEGIN CODE --------------

disp("BENCHMARK2 from POLAR")

% Parameter ---------------------------------------------------------------
% initial set
R0 = interval([0.7;0.7],[0.9;0.9]);
% time horizon
params.tFinal = 2.0;
params.R0 = polyZonotope(R0);

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.1; % tanh DNF
options.taylorTerms = 4;
options.zonotopeOrder = 200;
options.alg = 'lin';
options.tensorOrder = 3;
options.errorOrder = 10;
options.intermediateOrder = 50;

% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.bound_approx = true;
evParams.polynomial_approx = "lin";
evParams.add_approx_error_to_Grest = true;
evParams.remove_Grest = false;

% System Dynamics ---------------------------------------------------------

% open-loop system
f = @(x,u) [x(2) - x(1)^3; u(1)];
sys = nonlinearSys(f);

% load neural network controller
nn = NeuralNetwork.readONNXNetwork('nn_2_tanh.onnx');

% construct neural network controlled system
sys = neurNetContrSys(sys, nn, 0.2);

% Specification -----------------------------------------------------------

safeSet = 2 * interval(-ones(2, 1), ones(2, 1));
spec = specification(safeSet, 'safeSet');

% Simulation --------------------------------------------------------------

tic
simRes = simulateRandom(sys, params);
tSim = toc;
disp(['Time to compute random Simulations: ', num2str(tSim)]);

% Check Violation --------------------------------------------------------

tic
isVio = false;
for i = 1:length(simRes.x)
    x = simRes.x{i};
    for j=1:length(safeSet)
        isVio = isVio || ~all( ...
            (infimum(safeSet(j)) <= x(:, j)) & ...
            (x(:, j) <= supremum(safeSet(j))));
    end
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
    isVeri = true;
    for i = 1:length(R)
        R_i = R(i);
        for j = 1:length(R_i.timeInterval)
            isVeri = isVeri & safeSet.in(R_i.timeInterval.set{j});
        end
    end
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
rectangle('Position',[-0.3 -0.35 0.4 0.85])
%ss = plot(spec.set, [1, 2], 'FaceColor', [0, .8, 0]);
rs = plot(R, [1, 2], 'FaceColor', [.7, .7, .7]);
is = plot(R0, [1, 2], 'FaceColor', 'w', 'EdgeColor', 'k');
sims = plot(simRes, [1, 2], 'k');
xlabel('$x_1$ (distance)', 'interpreter', 'latex');
ylabel('$x_2\ (\dot{x_1})$', 'interpreter', 'latex')
xlim([-0.5, 1]);
ylim([-1.3, 1]);
rectangle('Position', [-0.3 -0.35 0.4 0.85])
legend([ss, rs, is, sims], "Safe Set", "Reachable Set", "Initial Set", "Simulations")

% figure;
% hold on;
% box on;
% ss = plot(spec.set, [3, 4], 'FaceColor', [0, .8, 0]);
% rs = plot(R, [3, 4], 'FaceColor', [.7, .7, .7]);
% is = plot(R0, [3, 4], 'FaceColor', 'w', 'EdgeColor', 'k');
% sims = plot(simRes, [3, 4], 'k');
% xlabel('$x_3$ (angle)', 'interpreter', 'latex');
% ylabel('$x_4\ (\dot{x_3})$', 'interpreter', 'latex')
% xlim([-2.5, 2.5]);
% ylim([-2.5, 2.5]);
% legend([ss, rs, is, sims], "Safe Set", "Reachable Set", "Initial Set", "Simulations")

% example completed
completed = 1;

%------------- END OF CODE --------------