function output = prior_density_Hybrid(model,alpha,mu_alpha,Sigma_alpha,a)
%% DESCRIPTION: 
% This function is used to compute the log of the prior density and
% gradient of the log prior density.

% INPUT: model = structure that contains model specifications
%        alpha = a matrix of transform random effects
%        mu_alpha = group-level mean 
%        Sigma_alpha = group-level covariance matrix
%        a = hyperparameter from the marginally non-informative prior of
%        the covariance matrix Sigma_alpha


% OUTPUT: is a structure that contains several fields
%     output.log = log of the prior density;
%     output.gradient = gradients of log prior density wrt alpha

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    J = model.num_subjects; 

%% Transform theta to C,Lambda and Sigma_alpha

    D_alpha = sum(model.dim); % number of random effects per participant
    v = model.prior_par.v_a + D_alpha - 1;
    Sigma_alpha_inv = Sigma_alpha\eye(D_alpha);

%% Compute gradients

% ------------- gradient wrt to alpha_1,...,alpha_n ----------------------
    A = Sigma_alpha_inv*(alpha - mu_alpha); % Tested compared with the inefficient way (for loop) below
    grad_alpha  = -A(:);

%---------------------- gradient wrt mu_alphaA -------------------------

    grad_mu = sum(A,2) - model.prior_par.cov\(mu_alpha-model.prior_par.mu); 
    
% ---------------------- gradient wrt log(a) ---------------------------

    grad_loga = -0.5*v - 1./(model.prior_par.A_d.^2) + 0.5./a + model.prior_par.v_a*diag(Sigma_alpha_inv)./a;
    
    log_Jacobian = sum(log(a)); %log Jacobian of log_a_d

    % ------------- This is the log prior when  mu_alpha ~ N(mu,cov) -------------
    output.log = -0.5*D_alpha*(J+1)*log(2*pi) - 0.5*(J + v + D_alpha +1)*logdet(Sigma_alpha) -... 
        0.5*trace((alpha-mu_alpha)'*A) - 0.5*logdet(model.prior_par.cov) - 0.5*(mu_alpha-model.prior_par.mu)'/model.prior_par.cov*(mu_alpha-model.prior_par.mu)+...
        0.5*v*(D_alpha*log(2*model.prior_par.v_a) - sum(log(a) ))  - 0.5*v*D_alpha*log(2)-...
        log_multigamma(v/2,D_alpha)- model.prior_par.v_a*sum(diag(Sigma_alpha_inv)./a)+...
        sum(inv_gamma_pdf( a,0.5*ones(D_alpha,1),1./(model.prior_par.A_d.^2),"true"))+log_Jacobian;

    output.grad = [grad_alpha; grad_mu; grad_loga]; % gradient of log(theta) wrt theta
end
