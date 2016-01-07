function [w,b] = hiddenlayer_build(n_in,n_out,activation)
    if activation == 'tanh'
        w_bound = sqrt(6./(n_in+n_out));
        w = random('Unif', -w_bound, w_bound, [n_out,n_in]);
        b = zeros(n_out,1);
    end
    
    fprintf('%s layer build complete\n',activation);
end