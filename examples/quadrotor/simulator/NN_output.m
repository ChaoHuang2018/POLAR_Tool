function [y] =  NN_output(x,name)


fid = fopen(name,'r');
tline = fgetl(fid);
no_of_inputs = str2num(tline);
%disp(no_of_inputs);

tline = fgetl(fid);
no_of_outputs = str2num(tline);
%disp(no_of_outputs);

tline = fgetl(fid);
no_of_hidden_layers = str2num(tline);
%disp(no_of_hidden_layers);

network_structure = zeros(no_of_hidden_layers+1,1);
for i = 1:no_of_hidden_layers
    tline = fgetl(fid);
    network_structure(i) = str2num(tline);
    %disp(network_structure(i));
end

activations = [];
for i = 1:no_of_hidden_layers+1
    tline = fgetl(fid);
    activations = [activations; convertCharsToStrings(tline)];
    % disp(tline);
end
% disp(activations)
network_structure(no_of_hidden_layers+1) = no_of_outputs;


weight_matrix = zeros(network_structure(1), no_of_inputs);
bias_matrix = zeros(network_structure(1),1);

% READING THE INPUT WEIGHT MATRIX
for i = 1:network_structure(1)
    for j = 1:no_of_inputs
        tline = fgetl(fid);
        weight_matrix(i,j) = str2num(tline);
    end
    tline = fgetl(fid);
    bias_matrix(i) = str2num(tline);
end
% disp(weight_matrix)

% Doing the input transformation
g = x;
% disp(size(weight_matrix))
% disp(size(g))
g = weight_matrix * g;
% disp(g)
g = g + bias_matrix;
% disp(g)
if strcmp( activations(1), "ReLU") == 1
    g = do_thresholding_relu(g);
end
if strcmp( activations(1), "sigmoid") == 1
    g = do_thresholding_sigmoid(g);
end
if strcmp( activations(1), "tanh") == 1
    g = do_thresholding_tanh(g);
end
if strcmp( activations(1), "Affine") == 1
end

% disp(g)

for i = 1:(no_of_hidden_layers)
    
    weight_matrix = zeros(network_structure(i+1), network_structure(i));
    bias_matrix = zeros(network_structure(i+1),1);

    % READING THE WEIGHT MATRIX
    for j = 1:network_structure(i+1)
        for k = 1:network_structure(i)
            tline = fgetl(fid);
            weight_matrix(j,k) = str2num(tline);
        end
        tline = fgetl(fid);
        bias_matrix(j) = str2num(tline);
    end
   
    % Doing the transformation
    g = weight_matrix * g;
    g = g + bias_matrix;
    
    if strcmp( activations(i+1), "ReLU") == 1
        g = do_thresholding_relu(g);
    end
    if strcmp( activations(i+1), "sigmoid") == 1
        g = do_thresholding_sigmoid(g);
    end
    if strcmp( activations(i+1), "tanh") == 1
        g = do_thresholding_tanh(g);
    end
    if strcmp( activations(i+1), "Affine") == 1
    end
    
    % disp("after activation: ")
    % disp(g)

end

offset = zeros(no_of_outputs,1);
tline = fgetl(fid);
for i = 1:no_of_outputs
    offset(i) = str2num(tline);
end
scale_factor = zeros(no_of_outputs,no_of_outputs);
tline = fgetl(fid);
for i = 1:no_of_outputs
    scale_factor(i,i) = str2num(tline);
end

y = g-offset;
y = scale_factor*y;

fclose(fid);

end