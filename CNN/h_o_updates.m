function [w,b,delta] = h_o_updates(learning_rate,w,b,in,out,des,activation)
    
    out_size = size(out);
    
    err = zeros(out_size);
    for i = [1 : 1 : out_size(2)]
        err(:,i) = out(:,i)-[zeros(des(i)-1,1); 1; zeros(10-des(i),1)];
    end
    
    if activation == 'sigmoid'
        delta = (err.*out.*(ones(out_size)-out));
    elseif activation == 'softmax'
        delta = err;
    end
    delta = delta/out_size(2);
    
    w = w - learning_rate*(delta*in.');
    b = b - learning_rate*sum(delta,2);
end

%diag(pred([1:size(pred,1)],y))