function output = inv_wishart_pdf(X,Psi,v)
% This is modified from the function 'iwpdf.m' ("VB PROJECT/Multivariate Gaussian")
% This function compute the log pdf of the Inverse-Wishart distribution
% output = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log(multigamma(v/2,p)) -...
%     0.5*(v+p+1)*logdet(x) - 0.5*trace(Psi/x);

%BE CAREFUL !!! Modified verson: input: C (NOT X) ,Psi,v where x = C*C';

[p,~] = size(Psi);
output = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log(multigamma(v/2,p)) -...
    0.5*(v+p+1)*logdet(X) - 0.5*trace(Psi/X);
end