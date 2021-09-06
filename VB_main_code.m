%% Hybrid VAFC for Hierarchical LBA models
%% Description:
% This is the main code to run the Hybrid VB for hierarchical LBA models.
% 
% Original paper: https://arxiv.org/abs/2102.06814
% 
% Author: Viet-Hung Dao (UNSW)
% 
% Email: viethung.unsw@gmail.com
%% Step 0: Preparation 
    addpath("Functions\") 
    addpath("Data\")   
    exp_name = "Mnemonic";
    m = 9; % the index of the chosen model
    load("Mnemonic.mat") % load the data
    matching_function_1 = "Matching_Mnemonic"; 
    matching_function_2 = "Matching_Gradients_Mnemonic";
    
    % ------------------ VB Tuning Parameter Setting ----------------------
    VB_settings.r = 15; % number of factors in VAFC
    VB_settings.max_iter = 10000; % the total number of iterations
    VB_settings.max_norm = 1000; 
    
    VB_settings.I = 10; % number of Monte Carlo samples used to estimate 
    VB_settings.window = 100;    
    VB_settings.patience_parameter = 50;                         
    VB_settings.learning_rate.v = 0.95;     
    VB_settings.learning_rate.eps = 10^(-7);   
    VB_settings.silent = "no"; % display the estimates at each iteration
    VB_settings.generated_samples = "yes"; 
%% Step 1: Model specification
if exp_name == "Your_experiment"
    model{m}.dim = [3 1 2 0 1];
    D_alpha = sum(model{m}.dim);
    model{m}.constraints = ["3" "1" "2" "0" "1"];
    model{m}.prior_par.mu = zeros(D_alpha,1); 
    model{m}.prior_par.cov = eye(D_alpha); 
    model{m}.prior_par.v_a = 2; 
    model{m}.prior_par.A_d = ones(D_alpha,1); 
else
    Model_specification
end                                                                        
%% Step 2: VB Algorithm  

    J = length(data); % number of subjects
    D_alpha = sum(model{m}.dim); % number of random effects per participant
    p1 = D_alpha*(J + 2); % dim(theta_1) = p1;

    model{m}.num_subjects = J;    
    model{m}.matching_function_1 = matching_function_1;
    model{m}.matching_function_2 = matching_function_2;
    model_name = ['c ~', model{m}.constraints(1),'| A ~',model{m}.constraints(2),...
        '| v ~',model{m}.constraints(3),'| s ~',model{m}.constraints(4),...
        '| t0 ~',model{m}.constraints(5)];
    disp([join(['Model ',num2str(m),' : ',model_name])]);
                                
%=========================== VB INITIALIZATION ==========================

%     VB_settings.r = 15 + sum(model{m}.index); % number of factors in VAFC
    r = VB_settings.r;
    if sum(model{m}.dim) < 10 
        epsilon = 1; % scale parameter of covariance matrix
        N_iter = 100; % number of iterations in PMwG to initialize VB
    elseif sum(model{m}.dim) < 20
        epsilon = 0.6;
        N_iter = 150; % number of iterations in PMwG to initialize VB
    else
        epsilon = 0.3;
        N_iter = 150; % number of iterations in PMwG to initialize VB
    end
    R = 10; % number of particles in PMwG
    initial_MCMC.alpha = randn(D_alpha,J);
    initial_MCMC.mu = randn(D_alpha,1); % the initial values for parameter \mu
    initial_MCMC.sig2 = iwishrnd(eye(D_alpha),D_alpha + 10); % the initial values for \Sigma
    initial_MCMC.a_d = 1./random('gam',1/2,1,D_alpha,1);       
    [MCMC_initial, initial_MCMC] = VB_adaptive_Initialization(model{m},data,initial_MCMC,N_iter,R,epsilon);
    initial = [reshape(MCMC_initial.alpha_store,D_alpha*J,1); MCMC_initial.mu_store; log(MCMC_initial.a_d_store)];
    lambda.mu = initial;
    lambda.B = zeros(p1,r)/r;    lambda.B = tril(lambda.B);
    lambda.d = 0.01*ones(p1,1);      
    VB_settings.initial = lambda;
    VB_settings.threshold = 1; % threshold for convergence criterion
    
%=========================== VB ALGORITHM ==========================
tic
    VB_results = Hybrid_VAFC(model{m},data,@Likelihood_Hybrid,@prior_density_Hybrid,VB_settings);   
CPUtime = toc;

%% Step 3: Extract the results
disp(['The running time is ',num2str(round(CPUtime/60,1)),' minutes'])      

plot(VB_results.LB_smooth)
title('Smoothed lower bound estimates')

disp('The best lambda is ')
VB_results.lambda
                                                                          
%% Step 4: Save the results

save('Mnemonic_model_1.mat');