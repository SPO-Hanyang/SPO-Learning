function [output] = convlayer(input,image_shape,filter_shape,w,b)

    out_size = [image_shape(1)-filter_shape(1)+1 image_shape(2)-filter_shape(2)+1 filter_shape(4) image_shape(4)];
    conv_size = [(image_shape(1)-filter_shape(1)+1)*(image_shape(2)-filter_shape(2)+1) filter_shape(4) image_shape(4)];
    conv_out = zeros(conv_size);

    % capture kernel size data from image
    % flatten captured data and weight
    % get result of each kernel with matrix multiple operation
    for i = [1 : 1 : image_shape(2)-filter_shape(2)+1]
        for j = [1 : 1 : image_shape(1)-filter_shape(1)+1]
            conv_out(((i-1)*(image_shape(2)-filter_shape(2)+1)+j),:,:) = reshape(w,[prod(filter_shape(1:3)) filter_shape(4)]).'*reshape(input(j:j+filter_shape(1)-1,i:i+filter_shape(2)-1,:,:),[prod(filter_shape(1:3)) image_shape(4)]);
        end
    end
    
    conv_out = conv_out + repmat(repmat(b,[1 conv_size(1)]).',[1 1 conv_size(3)]);
    
    output = tanh(conv_out);
    
    output = reshape(output,[out_size]);
   
end