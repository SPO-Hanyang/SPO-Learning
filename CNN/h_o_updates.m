function [w,b,delta] = h_o_updates(batch_size,learning_rate,w,b,in,out,des,activation)
    
    out_size = size(out,1);

    delta = zeros(out_size,batch_size);
    
    %theano
    for i = [1 : 1 : batch_size]
        del = zeros(out_size,1);
        true = des(i);
        err = out(:,i)-[zeros(true-1,1); 1; zeros(10-true,1)];
        if activation == 'sigmoid'
            del = (err.*out(:,i).*(ones(10,1)-out(:,i)));
        elseif activation == 'softmax'
            del = err;
        end
        del = del/batch_size;
        
        w = w - learning_rate*(del*in(:,i).');
        b = b - learning_rate*del;
        
        delta(:,i) = del;
    end
end