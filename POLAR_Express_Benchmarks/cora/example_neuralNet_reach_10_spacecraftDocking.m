function completed = example_neuralNet_reach_10_spacecraftDocking
% example_neuralNet_reach_10_spacecraftDocking - example of reachability analysis
%                                       for an neural network controlled
%                                       system
%
% Syntax:
%    completed = example_neuralNet_reach_10_spacecraftDocking()
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
% Written:      20-June-2022
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------

disp("BENCHMARK: Spacecraft Docking")

% Parameter ---------------------------------------------------------------

params.tFinal = 120;
params.R0 = polyZonotope(interval( ...
    [24/1000.0; 24/1000.0; -0.13776233054248638/0.5; -0.13776233054248638/0.5; 0.1948253562373095; 0.2697150717707441], ...
    [26/1000.0; 26/1000.0; -0.13776233054248638/0.5; -0.13776233054248638/0.5; 0.1948253562373095; 0.27552466108497276] ...
));

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.1;
options.taylorTerms = 4;
options.zonotopeOrder = 250;
% options.intermediateOrder = 100;
% options.errorOrder = 15;
% options.errorOrder3 = 15;
options.alg = 'lin';
options.tensorOrder = 2;

% polyZono.maxDepGenOrder = 100;
% polyZono.maxPolyZonoRatio = 0.001;
% polyZono.restructureTechnique = 'reduceFullGirard';
% options.polyZono = polyZono;

% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.bound_approx = true;
evParams.polynomial_approx = "lin";
evParams.add_approx_error_to_Grest = true;
evParams.remove_Grest = false;
% evParams.num_generators = 10000;

% System Dynamics ---------------------------------------------------------

% open-loop system
f = @dynamics_spacecraftDocking;
sys = nonlinearSys(f);

% load neural network controller
% [4, 256, 256, 4, 2]
nn = NeuralNetwork.readONNXNetwork('docking_tanh64x64_tanh.onnx');
% nn.evaluate(params.R0, evParams);
% nn.refine(2, "layer", "both", params.R0.c, true);


% construct neural network controlled system
sys = neurNetContrSys(sys, nn, 1);

% Specification -----------------------------------------------------------
% 
% v0 = 0.2;
% v1 = 2*0.001027;
% isSafe = @(x) (sqrt(x(3)^2 + x(4)^2)) <= (v0 + v1*sqrt(x(1)^2 + x(2)^2));

tic
R = reach(sys, params, options, evParams);
tComp = toc;
disp(['Time to compute Reachable Set: ', num2str(tComp)]);
% Simulation --------------------------------------------------------------
tic
simRes = simulateRandom(sys, params);
tSim = toc;
disp(['Time to compute random Simulations: ', num2str(tSim)]);

figure;
hold on;
box on;
% ss = plot(spec.set, [1, 2], 'FaceColor', [0, .8, 0]);
rs = plot(R, [5, 6], 'FaceColor', [.7, .7, .7]);
is = plot(params.R0, [5, 6], 'FaceColor', 'w', 'EdgeColor', 'k');
sims = plot(simRes, [5, 6], 'k');

xlabel('x');
ylabel('y');
legend([rs, is, sims], "Reachable Set", "Initial Set", "Simulations", Location="best")


% % Check Violation --------------------------------------------------------
% 
% tic
% isVio = false;
% for i = 1:length(simRes.x)
%     x_i = simRes.x{i};
%     for j = 1:size(x_i, 1)
%         isVio = isVio || ~isSafe(x_i(j, :)');
%         
%         if isVio
%             break;
%         end
%     end
% end
% tVio = toc;
% disp(['Time to check Violation in Simulations: ', num2str(tVio)]);


% if isVio 
%     disp("Result: VIOLATED")
%     R = [];
%     tComp = 0;
%     tVeri = 0;
% else
%     % Reachability Analysis -----------------------------------------------
% 
%     tic
%     R = reach(sys, params, options, evParams);
%     tComp = toc;
%     disp(['Time to compute Reachable Set: ', num2str(tComp)]);
% 
%     % Verification --------------------------------------------------------
% 
%     tic
%     Rend = R(end).timePoint.set{end};
%     % Rend = interval(Rend);
%     %isVeri = isSafe([Rend.inf(1:2);Rend.sup(3:4)]);
%     % x = [Rend.inf(1:2);Rend.sup(3:4)]';
%     
%     Q = {};
%     for i = 1:4
%         Q_i = zeros(4, 4);
%         Q_i(i, i) = 1;
%         Q{i} = Q_i;
%     end
% 
%     Rend_isSafe = [v1 0 0 0; 
%                    0 v1 0 0; 
%                    0 0 1 0; 
%                    0 0 0 1] * Rend;
%     Rend_isSafe = quadMap(Rend_isSafe, Q);
%     Rend_isSafe = [1 1 0 0; 0 0 1 1] * Rend_isSafe;
% 
%     evParams_isSafe = struct();
%     evParams_isSafe.bound_approx = false;
%     evParams_isSafe.reuse_bounds = true;
%     evParams_isSafe.polynomial_approx = "adaptive";
%     evParams_isSafe.add_approx_error_to_Grest = true;
%     evParams_isSafe.remove_Grest = true;
%     evParams_isSafe.num_generators = 10000;
%     evParams_isSafe.max_bounds = 1;
%     % evParams_isSafe.force_approx_lin_at = 0;
% 
%     nn_isSafe = NeuralNetwork({ ...
%         NNRootLayer()} ...
%     );
%     I = interval(Rend_isSafe, 'split');
%     nn_isSafe.layers{1}.l = max([0,0], I.inf');
%     nn_isSafe.layers{1}.u = I.sup';
%     nn_isSafe.layers{1}.order = 2;
%     Rend_isSafe = nn_isSafe.evaluate(Rend_isSafe, evParams_isSafe);
% 
% %     Rend_isSafe = [v1 0; 0 1] * Rend_isSafe;
%     Rend_isSafe = Rend_isSafe + [v0;0];
%     Rend_isSafe = interval(Rend_isSafe, 'split');
%     Rend_isSafe = [1,-1] * Rend_isSafe;
% 
%     % Rend_isSafe = reduce(Rend_isSafe, 'girard', 500);
%     isVeri = 0 <= Rend_isSafe.inf;
% 
% 
%     % isVeri = sqrt(Rend_isSafe.sup(2)) <= 0.2 + 2*0.001027 * sqrt(Rend_isSafe.inf(1));
%     % disp([sqrt(x(3)^2 + x(4)^2), 0.2 + 2*0.001027*sqrt(x(1)^2 + x(2)^2)]);
% 
%     tVeri = toc;
%     disp(['Time to check Verification: ', num2str(tVeri)]);
% 
%     if isVeri
%         disp('Result: VERIFIED');
%     else
%         disp('Result: UNKOWN')
%     end
% end
% 
% disp(['Total Time: ', num2str(tSim+tVio+tComp+tVeri)]);
% 
% % Visualization -----------------------------------------------------------
% disp("Plotting..")

% figure;
% hold on;
% box on;
% % gs = plot(goalSet, [1, 2], 'FaceColor', [0, .8, 0], 'EdgeColor', 'none');
% rs = plot(R, [1, 2], 'FaceColor', [.8, .8, .8], 'EdgeColor', 'none');
% is = plot(params.R0, [1, 2], 'FaceColor', 'w', 'EdgeColor', 'k');
% ss = plot(simRes, [1, 2], 'k');


% for dim=1
%     figure;
%     hold on;
%     box on;
%     
%     % gs = fill([0 0 params.tFinal, params.tFinal], [0.94, 1.06, 1.06, 0.94], 'g');
%     if ~isempty(R)
%         rs = plotOverTime(R, dim, 'FaceColor', [.8, .8, .8], 'EdgeColor', 'none');
%     else
%         rs = [];
%     end
%     % is = plotOverTime(params.R0, [1, 3], 'FaceColor', 'w', 'EdgeColor', 'k');
%     ss = plotOverTime(simRes, dim, 'k');
%     xlabel('time');
%     ylabel('x');
%     legend([rs, ss], "Reachable Set", "Simulations", Location="best")
% end

% example completed
completed = 1;

%------------- END OF CODE --------------
