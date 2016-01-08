function [cost] = negative_log_likelihood(pred,y)

    cost = -mean(log(diag(pred(y,[1:size(pred,2)]))));

end