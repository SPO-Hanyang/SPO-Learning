function [w,b] = convlayer_no_pooling_build(filter_shape)

    fan_in = prod(filter_shape(1:3));
    fan_out = prod(filter_shape(1:3));
    
    w_bound = sqrt(6./(fan_in+fan_out));
    
    w = random('Unif', -w_bound, w_bound, filter_shape);
    
    w = reshape(w,[prod(filter_shape(1:3)) filter_shape(4)]);
    
    b = zeros(filter_shape(4),1);
    
    fprintf('No Pooling Convolution layer build complete\n');

end