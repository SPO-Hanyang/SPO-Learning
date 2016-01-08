function [w,b,delta] = c_c_updates(learning_rate,filter_shape,w,b,in,out,prev_filter_shape,prev_w,prev_delta,activation)
    
    out_size = size(out);
    prev_delta_size = size(prev_delta);
    p_calc = zeros(out_size);
    temp_prev_w = reshape(prev_w,prev_filter_shape);
    
    % calculate delta from next node
    %{
    for i = [1 : 1 : prev_filter_shape(2)]
        for j = [1 : 1 : prev_filter_shape(1)]
            for m = [1 : 1 : prev_filter_shape(3)]
                p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) = p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) + sum(reshape(repmat(temp_prev_w(j,i,m,:),[prod(prev_delta_size(1:2)) prev_delta_size(4)]),prev_delta_size).*prev_delta,3);
            end
        end 
    end
    %}
    for i = [1 : 1 : prev_filter_shape(2)]
        for j = [1 : 1 : prev_filter_shape(1)]
            p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,:,:) = p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,:,:) + permute(reshape(reshape(temp_prev_w(j,i,:,:),[prev_filter_shape(3:4)])*reshape(permute(prev_delta,[3 1 2 4]),[prev_delta_size(3) prod(prev_delta_size(1:2))*prev_delta_size(4)]),[prev_filter_shape(3) prev_delta_size(1:2) prev_delta_size(4)]),[2 3 1 4]);
        end 
    end
    if(activation == 'tanh')
        phi = ones(out_size)-out.^2;
    end
    delta = phi.*p_calc;
    
    temp_w = zeros(filter_shape);
    
    for i = [1 : 1 : filter_shape(2)]
        for j = [1 : 1 : filter_shape(1)]
            temp_res=reshape(permute(in(j:j+out_size(1)-1,i:i+out_size(2)-1,:,:),[1 2 4 3]),[prod(out_size(1:2))*out_size(4) filter_shape(3)]).'*reshape(permute(delta,[1 2 4 3]),[prod(out_size(1:2))*out_size(4) filter_shape(4)]);
            temp_w(j,i,:,:) = temp_res;
        end
    end
    
    w = w - learning_rate*reshape(temp_w,[prod(filter_shape(1:3)) filter_shape(4)]);
    b = b - learning_rate*reshape(sum(sum(sum(delta,4),2),1),size(b));

end