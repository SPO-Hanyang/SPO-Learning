function [cost] = negative_log_likelihood(pred,y)

    cost = -mean(log(diag(pred([1:size(pred,1)],y))));

end