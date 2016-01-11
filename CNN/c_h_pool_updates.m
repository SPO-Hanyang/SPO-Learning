function [w,b,delta] = c_h_pool_updates(learning_rate,filter_shape,conv_stride,w,b,in,out,prev_w,prev_delta,activation,pool_size,pool_stride,pool_idx)
    
    out_size = size(out);
    
    % calculate delta from next node
    p_calc = prev_w.'*prev_delta;
    temp_calc = reshape(p_calc,[out_size(1:3) out_size(4)]);
    if(activation == 'tanh')
        phi = ones(out_size)-out.^2;
    end
    del = phi.*temp_calc;
    
    % rebuild delta to original size with max pooling index
    [delta,out_size] = rebuild_pool(del,size(del),pool_size,pool_stride,pool_idx);

    temp_w = zeros(filter_shape);
    
    for i = [1 : 1 : filter_shape(2)]
        for j = [1 : 1 : filter_shape(1)]
            temp_res=reshape(permute(in(j:conv_stride(1):j+out_size(1)-1,i:conv_stride(2):i+out_size(2)-1,:,:),[1 2 4 3]),[prod(out_size(1:2))*out_size(4) filter_shape(3)]).'*reshape(permute(delta,[1 2 4 3]),[prod(out_size(1:2))*out_size(4) filter_shape(4)]);
            temp_w(j,i,:,:) = temp_res;
        end
    end
    
    w = w - learning_rate*reshape(temp_w,[prod(filter_shape(1:3)) filter_shape(4)]);
    b = b - learning_rate*reshape(sum(sum(sum(delta,4),2),1),size(b));

end