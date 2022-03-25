function[h] = do_thresholding_relu(r)
    h = r;
    h(r<0) = 0;
end