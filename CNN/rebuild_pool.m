function [rebuild,origin_size] = rebuild_pool(pooled, pooled_size, pool_size, pool_stride, pool_idx)

    origin_size = [(pooled_size(1)-1)*pool_stride(1)+pool_size(1) (pooled_size(2)-1)*pool_stride(2)+pool_size(2) pooled_size(3) pooled_size(4)];
    rebuild = zeros(origin_size);
    temp_idx = reshape(pool_idx,[2 prod(pooled_size)]);
    temp_pooled = reshape(pooled,[prod(pooled_size) 1]);

    for i = [1 : 1 : pool_size(2)]
        for j = [1 : 1 : pool_size(1)]
            temp = zeros(size(temp_pooled));
            check_list = find(temp_idx(2,find(temp_idx(1,:)==j))==i);
            temp(check_list) = temp_pooled(check_list);
            rebuild(j:pool_stride(1):origin_size(1),i:pool_stride(2):origin_size(2),:,:) = rebuild(j:pool_stride(1):origin_size(1),i:pool_stride(2):origin_size(2),:,:)+reshape(temp,pooled_size);
        end
    end

end