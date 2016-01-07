function [output] = hiddenlayer(input,w,b,activation)
    net = w*input+repmat(b,[1 size(input,2)]);
    
    if activation == 'tanh'
        output = tanh(net);
    end
end