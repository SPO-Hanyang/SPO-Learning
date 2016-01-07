function [p_y_given_x,y_pred] = h_o_layer(input,w,b,activation)

    if activation == 'sigmoid'
        p_y_given_x = (1 ./ (1 + exp(-(w*input+repmat(b,[1 size(input,2)])))));
    elseif activation == 'softmax'
        p_y_given_x = softmax((w*input+repmat(b,[1 size(input,2)])));
    end
    
    [pred_max y_pred] = max(p_y_given_x);

end