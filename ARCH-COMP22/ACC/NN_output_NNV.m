function [y] =  NN_output_NNV(x)

load('controller_5_20_nnv.mat')
ss = size(act_fcns);
num_of_layers = ss(1);

g = x;

for i = 1:(num_of_layers)
    w_temp = W(1,i);
    w_mat = cell2mat(w_temp);
    size(w_mat)
    
    b_temp = b(i,1);
    b_mat = cell2mat(b_temp);
    size(b_mat)
    
    size(g)
    g = w_mat * g;
    g = g + b_mat;
    
    if i < num_of_layers
        g = do_thresholding_relu(g);
    end
    
end

y = g;

end