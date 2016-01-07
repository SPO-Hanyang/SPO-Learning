function [w,b,delta] = c_c_updates(batch_size,learning_rate,filter_shape,w,b,in,out,prev_filter_shape,prev_w,prev_delta,activation)
    
    out_size = size(out);
    prev_delta_size = size(prev_delta);
    p_calc = zeros(out_size);
    temp_prev_w = reshape(prev_w,prev_filter_shape);
    
    for i = [1 : 1 : prev_filter_shape(2)]
        for j = [1 : 1 : prev_filter_shape(1)]
            %for m = [1 : 1 : prev_filter_shape(3)]
               % p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) = p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) + sum(reshape(repmat(temp_prev_w(j,i,m,:),[prod(prev_delta_size(1:2)) prev_delta_size(4)]),prev_delta_size).*prev_delta,3);
             %  p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) = p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,m,:) + sum(reshape(repmat(temp_prev_w(j,i,m,:),[prod(prev_delta_size(1:2)) prev_delta_size(4)]),prev_delta_size).*prev_delta,3);
               
            %end
            p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,:,:) = p_calc(j:j+prev_delta_size(1)-1,i:i+prev_delta_size(2)-1,:,:) + sum( reshape(repmat(temp_prev_w(j,i,:,:),[prod(prev_delta_size(1:2)), prev_delta_size(4),prev_filter_shape(3)]),prev_delta_size) .* repmat(prev_delta,1,1,prev_filter_size(3)) ,3);
        end 
    end
    
    if(activation == 'tanh')
        pi = ones(out_size)-out.^2;
    end
    del = pi.*p_calc;
    delta = del;
    
    temp_w = zeros(filter_shape);
    
    for i = [1 : 1 : filter_shape(2)]
        for j = [1 : 1 : filter_shape(1)]
            for m = [1 : 1 : filter_shape(4)]
                for n = [1 : 1 : filter_shape(3)]
                    temp_res=in(j:j+out_size(1)-1,i:i+out_size(2)-1,n,:).*del(:,:,m,:);
                    temp_w(j,i,n,m) = sum(sum(sum(sum(temp_res,4),3),2),1);
                end
            end
        end
    end
    
    w = w - learning_rate*reshape(temp_w,[prod(filter_shape(1:3)) filter_shape(4)]);
    b = b - learning_rate*reshape(sum(sum(sum(del,4),2),1),size(b));

end