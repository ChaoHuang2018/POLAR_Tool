function read_nn_file(name)

file = fopen(name,'r');
file_data = fscanf(file,'%f');
no_of_inputs = file_data(1);
no_of_outputs = file_data(2);
no_of_hidden_layers = file_data(3);
network_structure = zeros(no_of_hidden_layers+1,1);
pointer = 4;

weights = {};
bias = {};
activations = ["tansig", "tansig", "tansig"];
for i = 1:no_of_hidden_layers
    network_structure(i) = file_data(pointer);
    pointer = pointer + 1;
end
network_structure(no_of_hidden_layers+1) = no_of_outputs;

weight_matrix = zeros(network_structure(1), no_of_inputs);
bias_matrix = zeros(network_structure(1),1);

% READING THE INPUT WEIGHT MATRIX
for i = 1:network_structure(1)
    for j = 1:no_of_inputs
        weight_matrix(i,j) = file_data(pointer);
        pointer = pointer + 1;
    end
    bias_matrix(i) = file_data(pointer);
    pointer = pointer + 1;
end

weights = [weights, transpose(weight_matrix)];
bias = [bias, transpose(bias_matrix)];

for i = 1:(no_of_hidden_layers)
    
    weight_matrix = zeros(network_structure(i+1), network_structure(i));
    bias_matrix = zeros(network_structure(i+1),1);

    % READING THE WEIGHT MATRIX
    for j = 1:network_structure(i+1)
        for k = 1:network_structure(i)
            weight_matrix(j,k) = file_data(pointer);
            pointer = pointer + 1;
        end
        bias_matrix(j) = file_data(pointer);
        pointer = pointer + 1;
    end
    
    weights = [weights, transpose(weight_matrix)];
    bias = [bias, transpose(bias_matrix)];

end

fclose(file);
save('net.mat', 'weights', 'bias', 'activations');

end