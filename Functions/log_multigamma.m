function y = log_multigamma(a,p)
%% Description:
% This function is to compute log of multigamma function. 
j = (1:1:p)';
y = p*(p-1)/4*log(pi) + sum(gammaln(a + (1-j)/2));
end

%% TESTED by
% comparing with log(multigamma(9,8)) and log_multigamma(9,8)