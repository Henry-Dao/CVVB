function output = p_Sigma_alpha(Sigma_alpha,v,v_a,Psi,alpha,mu_alpha,a)
%% DESCRIPTION: 
% This function is the conditional density of Sigma_alpha given the data
% and other parameters (refer to the Hybrid VB section in the paper)

% INPUT: Sigma_alpha = group-level covariance matrix
%        v = degress of freedom
%        v_a = group-level mean 
%        Psi = scale matrix
%        alpha = random effects (matrix)
%        mu_alpha = group-level mean
%        a = hyperparameter from the marginally non-informative prior of

% OUTPUT: is a structure that contains several fields
%     output.log = log of the density;
%     output.gradient = gradients of log density

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    [p,~] = size(Psi);
    Psi_inv = Psi\eye(p);   Sigma_alpha_inv = Sigma_alpha\eye(p);

    grad_logdet_alpha = 2*Psi_inv*(alpha-mu_alpha);
    grad_logdet_mu_alpha = -sum(grad_logdet_alpha,2); 
    grad_logdet_log_a = 2*v_a*diag(Psi_inv)./(-a);

    grad_trace_alpha = 2*Sigma_alpha_inv*(alpha-mu_alpha);
    grad_trace_mu_alpha = -sum(grad_trace_alpha,2); 
    grad_trace_log_a = 2*v_a*diag(Sigma_alpha_inv)./(-a);
    
    grad_alpha = 0.5*v*grad_logdet_alpha - 0.5*grad_trace_alpha;
    grad_mu_alpha = 0.5*v*grad_logdet_mu_alpha - 0.5*grad_trace_mu_alpha;
    grad_log_a = 0.5*v*grad_logdet_log_a - 0.5*grad_trace_log_a; 

    output.log = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log_multigamma(v/2,p) -...
        0.5*(v+p+1)*logdet(Sigma_alpha) - 0.5*trace(Psi*Sigma_alpha_inv);
    output.grad = [grad_alpha(:); grad_mu_alpha; grad_log_a];

end