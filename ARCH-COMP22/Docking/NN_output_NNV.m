function [y] =  NN_output_NNV(x)

load('model.mat')
ss = size(act_fcns);
num_of_layers = ss(1);

g = x;

for i = 1:(num_of_layers)
    w_temp = W(1,i);
    w_mat = cell2mat(w_temp);
    
    b_temp = b(1,i);
    b_mat = cell2mat(b_temp);
    
    g = w_mat * g;
    g = g + b_mat;
    
    if i == 2 || i == 3 || i == 5
        g = do_thresholding_tanh(g);
    end
end

y = g;

end