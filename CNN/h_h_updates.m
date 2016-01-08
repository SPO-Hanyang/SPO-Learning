function [w,b,delta] = h_h_updates(learning_rate,w,b,in,out,prev_w,prev_delta,activation)
    
    out_size = size(out);
    
    p_calc = prev_w.'*prev_delta;
    if(activation == 'tanh')
        pi = ones(out_size)-out.^2;
    end
    delta = (pi.*p_calc);
    
    w = w - learning_rate*(delta*in.');
    b = b - learning_rate*sum(delta,2);

end