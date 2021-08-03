function y = log_multigamma(a,p)

j = (1:1:p)';
y = p*(p-1)/4*log(pi) + sum(gammaln(a + (1-j)/2));
end
