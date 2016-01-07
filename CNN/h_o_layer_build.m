function [w,b] = h_o_layer_build(n_in,n_out,activation)

    w = zeros(n_out,n_in);
    b = zeros(n_out,1);
    fprintf('%s layer build complete\n',activation);

end