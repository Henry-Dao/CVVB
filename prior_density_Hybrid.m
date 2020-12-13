function output = prior_density_Hybrid(model,alpha,mu_alpha,a,Sigma_alpha,prior_par)
%% DESCRIPTION: If change prior_par then use general coding lines: 50,51; 57; 60-63; 69 - 71

% this density is modified based on 'prior_density_missing_Jacobian'
%   - line 54 & 56 (add missing Jacobian of Sigma_alpha)
%   - line 64 (add missing Jacobian of log_a_d)
%   - line 108, 109 and 115 ( add log Jacobians to the log density)
    J = model.num_subjects; 

%% Transform theta to C,Lambda and Sigma_alpha

    D_alpha = sum(model.index); % number of random effects per participant

%     Psi = 2*prior_par.v_a./a.*eye(D_alpha);
    v = prior_par.v_a + D_alpha - 1;
    Sigma_alpha_inv = Sigma_alpha\eye(D_alpha);

%% Compute gradients

% ------------- gradient wrt to alpha_1,...,alpha_n ----------------------

    % A = Sigma_alpha\(alpha - mu_alpha);
    A = Sigma_alpha_inv*(alpha - mu_alpha); % Tested compared with the inefficient way (for loop) below
    grad_alpha  = -A(:);
%     for j = 1:J 
%         A(:,j) = Sigma_alpha_inv*(alpha(:,j) - mu_alpha);
%     end


%---------------------- gradient wrt mu_alphaA -------------------------

    %grad_mu = sum(A,2) - prior_par.cov\(mu_alpha-prior_par.mu); 
    grad_mu = sum(A,2) - mu_alpha; %NOTE: this formula is simplified by the fact that par.sigma = I and par.mu = 0

    % sum(A,2) = Sigma_alpha_inv*sum(alpha - mu_alpha,2) TESTED !

% gradient wrt log(a)

    %grad_loga = -0.5*v*ones(p,1) - 1./(par.A.^2) + 0.5./a + par.v_alpha*diag(Sigma_alpha_inv)./a;
%     grad_loga1 = -0.5*v*ones(D_alpha,1)+ prior_par.v_a*(diag(Sigma_alpha_inv)./a); %gradient of p(Sigma_alpha) wrt loga;
%     grad_loga2 = -0.5 + (1./(prior_par.A_d.^2))./a; 
%     grad_loga =  grad_loga1 + grad_loga2;
    grad_loga = -0.5*v + 2*(diag(Sigma_alpha_inv)./a) - 0.5 + 1./a; % used when A_d = 1 and v_a = 2 (testd compared with above lines)

%------------------------ output of the function --------------------------
%
%-------------------------------------------------------------------------
%% OPTION 1: Compact, easy coding but computationally inefficient ( BUT HELPFUL TO CHECK !)

%     Sigma_alpha = C*C';
%     log_p_alpha = 0;
%     for j = 1:J
%         log_p_alpha = log_p_alpha + log(mvnpdf(alpha(:,j)',mu_alpha',Sigma_alpha));
%     end
%     
%     log_p_mu_alpha = log(mvnpdf(mu_alpha',prior_par.mu',prior_par.cov));
%     log_p_sigma_alpha = inv_wishart_pdf(Sigma_alpha,Psi,v);
%     log_p_a_d = sum(inv_gamma_pdf( a,0.5*ones(D_alpha,1),1./(prior_par.A_d.^2),"true"));
%     log_Jacobian = sum(log(a)); %log Jacobian of log_a_d
% 
%     log_prior = log_p_alpha + log_p_mu_alpha + log_p_sigma_alpha + log_p_a_d + log_Jacobian 

% --------------- OR A LITTLE BIT LESS INEFFICIENT ------------------------

%     log_p_alpha = -0.5*J*(D_alpha*log(2*pi)+ 2*logdet(C)) - 0.5*trace((alpha-mu_alpha)'*Sigma_alpha_inv*(alpha-mu_alpha));  
%     log_p_mu_alpha = -0.5*D_alpha*log(2*pi) - 0.5*logdet(prior_par.cov) - 0.5*(mu_alpha-prior_par.mu)'/prior_par.cov*(mu_alpha-prior_par.mu);
%     log_p_sigma_alpha = 0.5*v*logdet(Psi) - 0.5*v*D_alpha*log(2) - log(multigamma(v/2,D_alpha)) -(v+D_alpha+1)*logdet(C) - 0.5*trace(Psi*Sigma_alpha_inv);
%     log_p_a_d = sum(inv_gamma_pdf( a,0.5*ones(D_alpha,1),1./(prior_par.A_d.^2),"true"));
%     log_Jacobian = sum(log(a)); %log Jacobian of log_a_d
%     log_prior = log_p_alpha + log_p_mu_alpha + log_p_sigma_alpha + log_p_a_d + log_Jacobian

%% OPTION 2: COMPUTATIONAL EFFICIENT, BUT HARDER TO SEE ( CHECKED WITH OPTION 1 !!!)

log_Jacobian = sum(log(a)); %log Jacobian of log_a_d

% ----------- More efficient formulas:
%         logdet(Psi) = D_alpha*log(2*prior_par.v_a) - sum(theta.loga); 
%         trace(Psi*Sigma_alpha_inv) = 2*prior_par.v_a*sum(diag(Sigma_alpha_inv)./a)

% ------------- This is the log prior when  mu_alpha ~ N(mu,cov) -------------
% log_prior = -0.5*D_alpha*(J+1)*log(2*pi) - (J + v + D_alpha +1)*trace(C_star) -... 
%     0.5*trace((alpha-mu_alpha)'*A) - 0.5*logdet(prior_par.cov) - 0.5*(mu_alpha-prior_par.mu)'/prior_par.cov*(mu_alpha-prior_par.mu)+...
%     0.5*v*(D_alpha*log(2*prior_par.v_a) - sum(log(a) ))  - 0.5*v*D_alpha*log(2)-...
%     log(multigamma(v/2,D_alpha))- prior_par.v_a*sum(diag(Sigma_alpha_inv)./a)+...
%     sum(inv_gamma_pdf( a,0.5*ones(D_alpha,1),1./(prior_par.A_d.^2),"true"))+log_Jacobian

% NOTE: When mu_alpha ~ N(0,I), we have more efficient computation since:
%         logdet(prior_par.cov) = 0;    
%         (mu_alpha-prior_par.mu)'/prior_par.cov*(mu_alpha-prior_par.mu) = mu_alpha'*mu_alpha;

% ------------- This is the log prior when  mu_alpha ~ N(0,I) -------------
output.log = -0.5*D_alpha*(J+1)*log(2*pi) - 0.5*(J + v + D_alpha +1)*logdet(Sigma_alpha) -... 
    0.5*trace((alpha-mu_alpha)'*A) - 0.5*(mu_alpha'*mu_alpha)+...
    0.5*v*(D_alpha*log(2*prior_par.v_a) - sum(log(a) ))  - 0.5*v*D_alpha*log(2)-...
    log(multigamma(v/2,D_alpha))- prior_par.v_a*sum(diag(Sigma_alpha_inv)./a)+...
    sum(inv_gamma_pdf( a,0.5*ones(D_alpha,1),1./(prior_par.A_d.^2),"true"))+log_Jacobian;

output.grad = [grad_alpha; grad_mu; grad_loga]; %gradient of log(theta) wrt theta
end
