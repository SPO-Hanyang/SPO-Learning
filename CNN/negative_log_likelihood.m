function [cost] = negative_log_likelihood(pred,y)

    cost = -mean(log(pred(sub2ind(y,[1:size(pred,2)]))));

end