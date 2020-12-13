function output = p_Sigma_alpha(Sigma_alpha,v,v_a,Psi,alpha,mu_alpha,a)
% This is modified from the function 'iwpdf.m' ("VB PROJECT/Multivariate Gaussian")
% This function compute the log pdf of the Inverse-Wishart distribution
% output = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log(multigamma(v/2,p)) -...
%     0.5*(v+p+1)*logdet(x) - 0.5*trace(Psi/x);

%BE CAREFUL !!! Modified verson: input: C (NOT X) ,Psi,v where x = C*C';

[p,~] = size(Psi);
Psi_inv = Psi\eye(p);   Sigma_alpha_inv = Sigma_alpha\eye(p);

    grad_logdet_alpha = 2*Psi_inv*(alpha-mu_alpha);
    grad_logdet_mu_alpha = -sum(grad_logdet_alpha,2); % tested compared with below line
    % grad_logdet_mu_alpha = -2*Psi_inv*sum(alpha-mu_alpha,2);
    grad_logdet_log_a = 2*v_a*diag(Psi_inv)./(-a);

    grad_trace_alpha = 2*Sigma_alpha_inv*(alpha-mu_alpha);
    grad_trace_mu_alpha = -sum(grad_trace_alpha,2); % tested compared with below line
%     grad_trace_mu_alpha = -2*Sigma_alpha_inv*sum(alpha-mu_alpha,2);
    grad_trace_log_a = 2*v_a*diag(Sigma_alpha_inv)./(-a);
    
    grad_alpha = 0.5*v*grad_logdet_alpha - 0.5*grad_trace_alpha;
    grad_mu_alpha = 0.5*v*grad_logdet_mu_alpha - 0.5*grad_trace_mu_alpha;
    grad_log_a = 0.5*v*grad_logdet_log_a - 0.5*grad_trace_log_a; 

output.log = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log_multigamma(v/2,p) -...
    0.5*(v+p+1)*logdet(Sigma_alpha) - 0.5*trace(Psi*Sigma_alpha_inv);
% inv_wishart_pdf(Sigma_alpha,Psi,v);
%NOTE: using log(multigamma(v/2,p)) leads to Inf => use my function
%log_multigamma(v/2,p) (TESTED in simple cases)
output.grad = [grad_alpha(:); grad_mu_alpha; grad_log_a];


end