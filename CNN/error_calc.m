function [errors] = error_calc(pred,y)

    errors = 1-mean(pred==y);

end