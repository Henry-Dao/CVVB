function [output, initial] = VB_Lexical_adaptive_Initialization(model,data,initial_MCMC,N_iter,R,epsilon)

%% Load dataset
J = length(data.RT); % number of participants

%% Model Setting
num_choice = 2; % the number of choice
D_alpha = sum(model.index); % number of random effects per participant
D_G = D_alpha + D_alpha*(D_alpha+1)/2 + D_alpha; % D_G = dim(theta_G)total number of global parameters


                        % Prior setting
    prior_par.mu= zeros(D_alpha,1); %the prior for \mu_{\alpha}
    prior_par.cov = eye(D_alpha);%the prior for \Sigma_{\alpha}
    prior_par.v_a = 2; %the hyperparameters of the prior of \Sigma_{\alpha}
    prior_par.A_d = ones(D_alpha,1); %the hyperparameters of the prior of \Sigma_{\alpha}
% For easy reading code: 
%     v_a = prior_par.v_a;    A_d = prior_par.A_d;
%% MCMC Setting

                % Set the MCMC tuning parameters
% R = 10; % number of particles in the conditional Monte Carlo algorithm
N_burn = N_iter-50+1;     % the burn in iterations
N_adapt = 0;
N_sampling = 0;    % we take the last 10000 draws out of 12000 draws
% N_iter = N_burn + N_adapt + N_sampling;     % the maximum total number of iterations
count = 1;
better_proposal = false;
% epsilon = 0.8; %% the scaling parameter for the proposal during burn in and initial adaptation.

                % Allocation memory for variables storing MCMC draws
mu_store = zeros(D_alpha,N_iter);
vech_C_store = zeros(D_alpha*(D_alpha+1)/2,N_iter);  % Sigma = C'*C;
vech_C_star_store = zeros(D_alpha*(D_alpha+1)/2,N_iter); % C^* is tranformed !
a_d_store = zeros(D_alpha,N_iter);
% theta_G_store = zeros(D_G,N_iter); % matrix stores draws of transformed global parameters: (mu, vech(C*),loga)
alpha_store = zeros(D_alpha,J,N_iter); % 3-D matrix that stores random effects draws

mean_prop = zeros(D_G,J); % allocation for proposal mean for the random effects, stored in columns
cov_prop = zeros(D_G,D_G,J); % allocation for proposal covariance matrix for the random effects 

                        % Initialization

alpha = initial_MCMC.alpha;
theta_G.mu = initial_MCMC.mu; %the initial values for parameter \mu
theta_G.sig2 = initial_MCMC.sig2; % the initial values for \Sigma
theta_G.a_d = initial_MCMC.a_d;
% a_d = theta_G.a_d; %initial values for a_{1},...,a_{7}


%[theta_latent] = LBA_MC_v1(data,param,num_subjects,num_trials,num_particles); %obtain initial values of the random effects.


%% THE MCMC - Particles Metropolis within Gibbs
% parpool(48) %number of processors available to be used.
t = 1;
count = 1;
while t<=N_iter
%     tic
    
% ------------------- Sample \mu|rest in Gibbs step --------------------
    
    var_mu = (J*inv(theta_G.sig2) + inv(prior_par.cov))\eye(D_alpha); % General formula would be: var_mu = inv(J/theta.sig2 + inv(prior_mu_sig2));
    mean_mu = var_mu*( theta_G.sig2\sum(alpha,2) + prior_par.cov\prior_par.mu);
    theta_G.mu = mvnrnd(mean_mu,var_mu)';
    
% ------------------ Sample \Sigma|rest in Gibbs step --------------------

    k_a = prior_par.v_a + D_alpha - 1 + J;
    cov_temp = zeros(D_alpha);
    for j=1:J
        cov_temp = cov_temp + (alpha(:,j)-theta_G.mu)*(alpha(:,j)-theta_G.mu)';
    end
    B_a = 2*(prior_par.v_a)*diag(1./theta_G.a_d) + cov_temp;
    theta_G.sig2 = iwishrnd(B_a,k_a);    
    theta_sig2_inv = inv(theta_G.sig2);

% -------------- Sample a_{1},...,a_{7}|rest in Gibbs step ----------------

    theta_G.a_d = 1./gamrnd((prior_par.v_a + D_alpha)/2, 1./(prior_par.v_a*diag(theta_sig2_inv) + (1./prior_par.A_d).^2 ));
 

% ----------------- Sample alpha_j|rest in Gibbs step ---------------------- 

   
 % conditional Monte Carlo algorithm to update the random effects   
 
   [alpha] = LBA_CMC_Lexical(model,data,theta_G,alpha,J,D_alpha,R,mean_prop,cov_prop,better_proposal,epsilon);
%     sig2 = theta_G.sig2;
%     mu = theta_G.mu;
%     parfor j=1:J
%         
%         w_mix = 0.5; % setting the weights of the mixture in the burn in and initial sampling stage.
%         if rand()< w_mix                  
%             alpha(:,j) = mvnrnd(alpha(:,j)', epsilon*sig2)';
%         else
%             alpha(:,j) = mvnrnd(mu',sig2)';
%         end
%     end      
%   --------------------- storing the MCMC draws  -------------------------    
    
    % storing the global parameters
    mu_store(:,t) = theta_G.mu;
       
    chol_sig2 = chol(theta_G.sig2,'lower');
    C_star = chol_sig2;     C_star(1:D_alpha+1:end) = log(diag(chol_sig2));
    vech_C_store(:,t) = vech(chol_sig2);
    vech_C_star_store(:,t) = vech(C_star);
    
    a_d_store(:,t) = theta_G.a_d;
    

    % storing random effects    
    alpha_store(:,:,t) = alpha;
    
     t=t+1;   
end

initial.alpha = alpha;
initial.mu = theta_G.mu;
initial.a_d = theta_G.a_d;
initial.sig2 = theta_G.sig2;

output.mu_store = mean(mu_store(:,N_burn:end),2);
output.vech_C_star_store = mean(vech_C_star_store(:,N_burn:end),2);
output.a_d_store = mean(a_d_store(:,N_burn:end),2);
output.alpha_store = mean(alpha_store(:,:,N_burn:end),3);

%% save the final output in the standard MCMC output structure
end
