function output = pdf_c(t,b,A,v,s,tau,gradient)
%DECRIPSTION: created at 1:41 am 14/06/2020
% NUMERICALLY CHECKED !!!
% This is the function f_c(t)
%INPUT: t = response time (vector)
%       v = IS A VECTOR
%       z = (b,A,v,tau)
%OUTPUT: .value = the value of f_c(t);
%        .grad_b = gradient of  log(f_c(t)) wrt b

t_diff = t-tau;
%shortcut computation (see paper page 5 for notation x,y,z,u)
w1 = (b - t_diff.*v)./(s.*t_diff); w2 = A./(s.*t_diff);
x = normcdf(w1-w2); x1 = normpdf(w1-w2); 
z = normcdf(w1);  z1 = normpdf(w1); 

%% This from David's code 'LBA_tpdf.m'to avoid numerical error ( David confirmed with Scott)
    id = x>0.999;
    x(id,1) = 0.999;
    id = x<=0.001;
    x(id,1) = 0.001;
    
    id = z>0.999;
    z(id,1) = 0.999;
    id = z<0.001;
    z(id,1) = 0.001;

%%
f_value = (1./A).*( v.*(z-x) + s.*(x1  - z1) ); %the value of f_c(t)

ind_fc = f_value<=10^-50;  % Also, from David code ``LBA_n1PDF_reparam_real.m''(confirmed with Scott) to avoid numerically zero
f_value(ind_fc) = 10^-50;  % Also, from David code ``LBA_n1PDF_reparam_real.m''(confirmed with Scott) to avoid numerically zero

output.value = f_value;

%Compute the partial derivatves
if (gradient == "true")
    x2 = phi_prime(w1-w2);  z2 = phi_prime(w1);
    
    grad_b = (1./A).*(v.*(z1-x1) + s.*(x2- z2) )./(t_diff.*s);
    grad_A = -(1./A).*( f_value + (-v.*x1 + s.*x2)./(s.*t_diff));
    grad_v = (1./A).*(z-x) + (1./A).*( v.*(z1-x1) + s.*(x2-z2) )./(-s);
    grad_s = (1./A).*( (s.*x2 - v.*x1).*((w2-w1)./s) + (v.*z1 - s.*z2).*(-w1./s) +x1 - z1);
    grad_tau = (1./A).*( (-v.*x1 + s.*x2).*(b-A)./(s.*t_diff.^2) + (v.*z1-s.*z2).*b./(s.*t_diff.^2)  );

    
    output.grad_b = grad_b./f_value;
    output.grad_A = grad_A./f_value;
    output.grad_v = grad_v./f_value;
    output.grad_s = grad_s./f_value;
    output.grad_tau = grad_tau./f_value;
end
% output.grad = [grad_b, grad_A, grad_v, grad_s, grad_tau];
%NOTE: 1. this is not the correct gradient. The correct gradient is to set one
% grad_v equal 0 (which grad_v depends on the observed response choice)
%      2. since t is a vector then gradients are stored in columns 
end


