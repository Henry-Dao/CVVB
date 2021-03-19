function y = multigamma(a,p)
%INPUT: G_p(a);
b = 1;
for i=1:p
    b = b*gamma(a + 0.5*(1-i));
end
y = pi^(0.25*p*(p-1))*b;
end