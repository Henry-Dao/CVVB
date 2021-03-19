function [y] = inv_gamma_pdf(x,shape,scale,log_value)
%INPUT: x, scale and shape are vectors of same length (column vectors)
%OUTPUT: y = vector of log pdf of inverse gamma
    a = shape;  b = scale;
    if log_value == "true"
        y = a.*log(b) - log(gamma(a)) - (a+1).*log(x) - b./x;
    else
        y = (b.^a./gamma(a)).*x.^(-a-1).*exp(-b./x);
    end
end