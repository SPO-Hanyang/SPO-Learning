function [w,b,delta] = c_h_updates(learning_rate,filter_shape,w,b,in,out,prev_w,prev_delta,activation)
    
    out_size = size(out);
    
    % calculate delta from next node
    p_calc = prev_w.'*prev_delta;
    temp_calc = reshape(p_calc,[out_size(1:3) out_size(4)]);
    if(activation == 'tanh')
        phi = ones(out_size)-out.^2;
    end
    delta = phi.*temp_calc;
    
    temp_w = zeros(filter_shape);
    
    for i = [1 : 1 : filter_shape(2)]
        for j = [1 : 1 : filter_shape(1)]
            for m = [1 : 1 : filter_shape(4)]
                temp_res=in(j:j+out_size(1)-1,i:i+out_size(2)-1,:,:).*repmat(delta(:,:,m,:),[1 1 filter_shape(3) 1]);
                temp_w(j,i,:,m) = sum(sum(sum(temp_res,4),2),1);
            end
        end
    end
    
    w = w - learning_rate*reshape(temp_w,[prod(filter_shape(1:3)) filter_shape(4)]);
    b = b - learning_rate*reshape(sum(sum(sum(delta,4),2),1),size(b));

end