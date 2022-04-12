function[h] = do_thresholding_sigmoid(r)
    h = 1 ./ (1+exp(-r));
end