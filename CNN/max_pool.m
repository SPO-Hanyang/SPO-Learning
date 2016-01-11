function [pooled,idx] = max_pool(input, input_size, pool_size, pool_stride)

    pooled = zeros((input_size(1)-pool_size(1))/pool_stride(1)+1,(input_size(2)-pool_size(2))/pool_stride(2)+1,input_size(3),input_size(4));
    idx = zeros(2,(input_size(1)-pool_size(1))/pool_stride(1)+1,(input_size(2)-pool_size(2))/pool_stride(2)+1,input_size(3),input_size(4));
    temp_pooled = -99*ones(prod(size(pooled)),1);
    temp_idx = zeros(2,prod(size(idx))/2);

    for i = [1 : 1 : pool_size(2)]
        for j = [1 : 1 : pool_size(1)]
            temp = input(j:pool_stride(1):input_size(1),i:pool_stride(2):input_size(2),:,:);
            [temp_pooled max_check] = max([temp_pooled reshape(temp,[prod(size(temp)) 1])],[],2);
            ch_idx = find(max_check == 2);
            temp_idx(1,ch_idx) = j;
            temp_idx(2,ch_idx) = i;
        end
    end
    
    pooled = reshape(temp_pooled,size(pooled));
    idx = reshape(temp_idx,size(idx));

end