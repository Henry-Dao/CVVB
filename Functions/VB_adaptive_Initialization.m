function [output, initial] = VB_adaptive_Initialization(model,data,initial_MCMC,N_iter,R,epsilon)
%% DESCRIPTION: 
% This function is used to carefully choose an initial value for VB (by
% running PMwG with a small number of iterations (R = 50 to 100).

% INPUT: model = structure that contains model specifications
%        data = the data (in the correct format, see the user manual)
%        initial_MCMC = initial values for PMwG
%        N_iter = total number of iterations. 
%        R = number of particles
%        epsilon = scale parameter
% OUTPUT: .ouput = average of the latest MCMC draws (the first half of MCMC draws are thrown away as burn-in) ;
%         .initial = store the last MCMC draw

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    J = model.num_subjects; % number of participants
    D_alpha = sum(model.dim); % number of random effects per participant
    Matching_Function_1 = str2func(model.matching_function_1);
%---------------- Allocate memory for variables storing MCMC draws --------
    mu_store = zeros(D_alpha,N_iter);
    vech_C_star_store = zeros(D_alpha*(D_alpha+1)/2,N_iter); % C^* is tranformed !
    a_d_store = zeros(D_alpha,N_iter);
    alpha_store = zeros(D_alpha,J,N_iter); % 3-D matrix that stores random effects draws

%----------------------------- Initialization -----------------------------

    alpha = initial_MCMC.alpha;
    theta_G.mu = initial_MCMC.mu; %the initial values for parameter \mu
    theta_G.sig2 = initial_MCMC.sig2; % the initial values for \Sigma
    theta_G.a_d = initial_MCMC.a_d;

%% THE MCMC - Particles Metropolis within Gibbs
t = 1;
while t<=N_iter    
% ------------------- Sample \mu|rest in Gibbs step --------------------
    
    var_mu = (J*inv(theta_G.sig2) + inv(model.prior_par.cov))\eye(D_alpha); % General formula would be: var_mu = inv(J/theta.sig2 + inv(prior_mu_sig2));
    mean_mu = var_mu*( theta_G.sig2\sum(alpha,2) + model.prior_par.cov\model.prior_par.mu);
    theta_G.mu = mvnrnd(mean_mu,var_mu)';
    
% ------------------ Sample \Sigma|rest in Gibbs step --------------------

    k_a = model.prior_par.v_a + D_alpha - 1 + J;
    cov_temp = zeros(D_alpha);
    for j=1:J
        cov_temp = cov_temp + (alpha(:,j)-theta_G.mu)*(alpha(:,j)-theta_G.mu)';
    end
    B_a = 2*(model.prior_par.v_a)*diag(1./theta_G.a_d) + cov_temp;
    theta_G.sig2 = iwishrnd(B_a,k_a);    
    theta_sig2_inv = inv(theta_G.sig2);

% -------------- Sample a_{1},...,a_{7}|rest in Gibbs step ----------------

    theta_G.a_d = 1./gamrnd((model.prior_par.v_a + D_alpha)/2, 1./(model.prior_par.v_a*diag(theta_sig2_inv) + (1./model.prior_par.A_d).^2 ));
 
% ----------------- Sample alpha_j|rest in Gibbs step ---------------------   
    parfor j=1:J
        n_j = length(data{j}.RT);    
    % ----- (step 1): Generate alpha_j from the proposal distribution -----
    %
    % ---------------------------------------------------------------------   
        alpha_j_k = alpha(:,j); % the set of random effects from previous iteration of MCMC for conditioning. 
        w_mix = 0.5; % setting the weights of the mixture in the burn in and initial sampling stage.
        u = rand(R,1);
        id1 = (u<w_mix);
        n1 = sum(id1);
        n2 = R-n1;
        chol_covmat = chol(theta_G.sig2,'lower');
        rnorm1 = alpha_j_k + epsilon.*chol_covmat*randn(D_alpha,n1);
        rnorm2 = theta_G.mu + chol_covmat*randn(D_alpha,n2);
        alpha_j_R = [rnorm1 rnorm2];  % alpha_j_R = [alpha_j^1, ... alpha_j^R]. Particles alpha_j^r are stores in colums of matrix alpha_j_R
    
        alpha_j_R(:,1) = alpha_j_k;
        
    % -------------- (step 2): Compute the importance weights  ----------------
    %
    % -------------------------------------------------------------------------    
  
        % Duplicate data (y_j) and stack them into a column vector

        RT_j = repmat(data{j}.RT,R,1);
        RE_j = repmat(data{j}.RE,R,1); % RE = 1 (error) 2 (correct)

        % Match the random effects with the observations
        z_j = Matching_Function_1(model,alpha_j_R,data{j});
        
        FUNC = LBA_pdf(RE_j,RT_j,z_j{1},z_j{2},z_j{3},z_j{4},z_j{5},"false");

        log_LBA_j = FUNC.log_element_wise;
        lw_reshape = reshape(log_LBA_j,n_j,R);
        logw_first = sum(lw_reshape)'; 

        % Computing the log of p(\alpha|\theta) and density of the proposal for
        
        logw_second = log(mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2));
        logw_third = log(w_mix.*mvnpdf(alpha_j_R',alpha_j_k',(epsilon^2).*theta_G.sig2)+...
            (1-w_mix).*mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2));
        logw = logw_first + logw_second - logw_third;

        max_logw = max(logw);
        weight = exp(logw-max_logw);	
        weight = weight./sum(weight);
   
        Nw = length(weight);

        if (sum(weight>0)>0) 
            ind = randsample(Nw,1,true,weight);
            alpha(:,j) = alpha_j_R(:,ind);
        end
    end   
% --------------------- storing the MCMC draws  -------------------------    
    
    mu_store(:,t) = theta_G.mu;
    chol_sig2 = chol(theta_G.sig2,'lower');
    C_star = chol_sig2;     C_star(1:D_alpha+1:end) = log(diag(chol_sig2));
    vech_C_star_store(:,t) = vech(C_star);   
    a_d_store(:,t) = theta_G.a_d;
    alpha_store(:,:,t) = alpha;
    
    t = t+1;   
end
%% save the final output in the standard MCMC output structure
    N_burn = ceil(N_iter/2);     % the burn-in iterations
    
    initial.alpha = alpha;
    initial.mu = theta_G.mu;
    initial.a_d = theta_G.a_d;
    initial.sig2 = theta_G.sig2;

    output.mu_store = mean(mu_store(:,N_burn:end),2);
    output.vech_C_star_store = mean(vech_C_star_store(:,N_burn:end),2);
    output.a_d_store = mean(a_d_store(:,N_burn:end),2);
    output.alpha_store = mean(alpha_store(:,:,N_burn:end),3);
end
