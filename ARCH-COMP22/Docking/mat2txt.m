load('model.mat');
num_of_input = size(W{1}, 1);
num_of_output = size(W{end}, 2);
num_of_hidden_layers = size(W, 2) - 1;

num_of_hidden_neurons = [];
activations = [];

fileID = fopen('model_POLAR', 'w');

fprintf(fileID, '%d', num_of_input);
fprintf(fileID, '\n');
fprintf(fileID, '%d', num_of_output);
fprintf(fileID, '\n');
fprintf(fileID, '%d', num_of_hidden_layers);
fprintf(fileID, '\n');

for idxLayer = 1:num_of_hidden_layers
    num_of_hidden_neurons = [num_of_hidden_neurons, size(W{idxLayer}, 1)];
    fprintf(fileID, '%d', size(W{idxLayer}, 1));
    fprintf(fileID, '\n');
end

for idxLayer = 1:num_of_hidden_layers + 1
    if act_fcns(idxLayer, :) == 'linear'
        activation = "Affine";
    elseif act_fcns(idxLayer, :) == 'tanh  '
        activation = "tanh";    
    end
    fprintf(fileID, '%s', activation);
    fprintf(fileID, '\n');
    activations = [activations, activation];
end

for idxLayer = 1:num_of_hidden_layers + 1
    weight = W{idxLayer};
    for row = 1:size(weight, 1)
        % write weights
        for col = 1:size(weight, 2)
            fprintf(fileID, '%f', weight(row, col));
            fprintf(fileID, '\n');
        end
        % write biases
        fprintf(fileID, '%f', b{idxLayer}(row));
        fprintf(fileID, '\n');
    end
end

fprintf(fileID, '%d', 0);
fprintf(fileID, '\n');
fprintf(fileID, '%d', 1);

fclose(fileID);
