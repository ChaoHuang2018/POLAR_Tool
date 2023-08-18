function completed = example_neuralNet_reach_08_attitudeControl
% example_neuralNet_reach_08_attitudeControl - example of reachability analysis
%                                       for an neural network controlled
%                                       system
%
% Syntax:
%    completed = example_neuralNet_reach_08_attitudeControl()
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

disp("BENCHMARK: Attitude Control")

% Parameter ---------------------------------------------------------------

params.tFinal = 3;
params.R0 = polyZonotope(interval( ...
    [-0.45, -0.55, 0.65, -0.75, 0.85, -0.65], ...
    [-0.44, -0.54, 0.66, -0.74, 0.86, -0.64] ...
));

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.1;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
options.alg = 'lin';
options.tensorOrder = 2;

% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.bound_approx = true;
evParams.polynomial_approx = "lin";
evParams.add_approx_error_to_Grest = true;
evParams.remove_Grest = false;

% System Dynamics ---------------------------------------------------------

% open-loop system
f = @dynamics_attitudeControl;
sys = nonlinearSys(f);

% load neural network controller
% [4, 500, 2]
nn = NeuralNetwork.readONNXNetwork('attitude_control_3_64_torch.onnx');

% construct neural network controlled system
sys = neurNetContrSys(sys, nn, 0.1);

% Specification -----------------------------------------------------------

unsafeSet = interval( ...
    [-0.2;-0.5;0;-0.7;0.7;-0.4], ...
    [0;-0.4;0.2;-0.6;0.8;-0.2] ...
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
    isVio = isVio || unsafeSet.in(simRes.x{i}(end, :)');
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
    Rend = interval(Rend);

    isVeri = ~isIntersecting(Rend, unsafeSet);
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
disp(R)
% gs = plot(goalSet, [1, 2], 'FaceColor', [0, .8, 0], 'EdgeColor', 'none');
rs = plot(R, [1, 2], 'FaceColor', [.8, .8, .8], 'EdgeColor', 'none');
is = plot(params.R0, [1, 2], 'FaceColor', 'w', 'EdgeColor', 'k');
ss = plot(simRes, [1, 2], 'k');
xlabel('x');
ylabel('y');
% title('timestep0.005')
legend([rs, is, ss], "Reachable Set", "Initial Set", "Simulations", Location="best")

% for w=1:3
%     figure;
%     hold on;
%     box on;
%     R0 = project(params.R0, w);
%     R0 = cartProd(polyZonotope(0, [], [], []), R0);
%     
%     us = plot(cartProd(interval(params.tFinal, params.tFinal), project(unsafeSet, w)), [1, 2], 'FaceColor', [.8, 0, 0]);
%     alpha(us,.5)
%     rs = plotOverTime(R, w, 'FaceColor', [.8, .8, .8], 'EdgeColor', 'none');
%     is = plot(R0, [1, 2], 'FaceColor', 'w', 'EdgeColor', 'k');
%     ss = plotOverTime(simRes, w, 'k');
%     xlabel('time');
%     ylabel(sprintf("\\omega_%d", w));
%     legend([us, rs, is, ss], "Unsafe Set", "Reachable Set", "Initial Set", "Simulations", Location="best")
% end

% example completed
completed = 1;

%------------- END OF CODE --------------
