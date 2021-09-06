%% K-fold CVVB for Selecting between Hierarchical LBA models
%% Description:
% This is the main code to run the K-fold CVVB for comparing hierarchical LBA 
% models.
% 
% Original paper: https://arxiv.org/abs/2102.06814
% 
% The VB method used in CVVB is the Hybrid VB (refer to the paper)
% 
% Author: Viet-Hung Dao (UNSW)
% 
% Email: viethung.unsw@gmail.com

%% Step 0: Preparation   
    addpath("Functions/") % to load all the required functions
    addpath("Data/") % to load the data  
    load("Mnemonic.mat")
    exp_name = "Mnemonic"; % for model coding. 
    matching_function_1 = "Matching_Mnemonic";
    matching_function_2 = "Matching_Gradients_Mnemonic";
    K = 5; % number of folds in CV
    
    % ------------------ VB Tuning Parameter Setting ----------------------
    VB_settings.r = 15; % number of factors in VAFC
    VB_settings.max_iter = 10000; % the total number of iterations
    VB_settings.max_norm = 1000; 
    
    VB_settings.I = 10; % number of Monte Carlo samples used to estimate 
    VB_settings.window = 100;    
    VB_settings.patience_parameter = 50;                         
    VB_settings.learning_rate.v = 0.95;     
    VB_settings.learning_rate.eps = 10^(-7);   
    VB_settings.silent = "yes"; % display the estimates at each iteration
    VB_settings.generated_samples = "no"; 
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
%% Step 2: Randomly partition the data into K folds
    J = length(data); % number of participants
    train_ind = cell(K,J);  test_ind = cell(K,J);
    Fieldlist = fieldnames(data{1});
    for j = 1:J
        n_j = length(data{j}.RT);
        RandInd = randperm(n_j);
        fold_size = floor(n_j/K);
        for k = 1:K-1
            test_ind{k,j} = RandInd((k-1)*fold_size + 1 : k*fold_size);
            train_ind{k,j} = setdiff(RandInd,test_ind{k,j});
        end
        test_ind{K,j} = RandInd(k*fold_size + 1 : end);
        train_ind{K,j} = setdiff(RandInd,test_ind{K,j});
    end
    
%% Step 3: K-fold CVVB Algorithm  

    ELPD_CVVB = zeros(K,M); % Store all the ELPD estimates
    VB_results = cell(K,M); % Store all results of VB
    total_time = 0;
    for m = 1:M
        tic
        D_alpha = sum(model{m}.dim); % number of random effects per participant
        p1 = D_alpha*(J + 2); % dim(theta_1) = p1;

        model{m}.num_subjects = J;    
        model{m}.matching_function_1 = matching_function_1;
        model{m}.matching_function_2 = matching_function_2;
        model_name = ['c ~', model{m}.constraints(1),'| A ~',model{m}.constraints(2),'| v ~',model{m}.constraints(3),'| s ~',model{m}.constraints(4),'| t0 ~',model{m}.constraints(5)];
        disp([join(['Model ',num2str(m),' : ',model_name])]);

    % ----------------- VB Initialization  -----------   

%         VB_settings.r = 15 + sum(model{m}.dim); % number of factors in VAFC
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

    %=========================== CVVB FOR THE CURRENT MODEL =======================
        k = 1;
        while k<=K
            %(1) Split the dataset into training and test sets
            train_data = cell(J,1);     test_data = cell(J,1);
            for j = 1:J               
                for iField = 1:numel(Fieldlist)
                    Field = Fieldlist{iField};
                    train_data{j}.(Field) = data{j}.(Field)(train_ind{k,j});
                    test_data{j}.(Field) = data{j}.(Field)(test_ind{k,j});
                end
            end
                
            if k==1
                convergence = "no";
                count = 1;
                VB_settings.threshold = 1; % threshold for convergence criterion
                while convergence == "no" && count <=5
                    VB_settings.threshold = 1 + (count>3)*(count-3); % threshold for convergence criterion
                    initial_MCMC.alpha = randn(D_alpha,J);
                    initial_MCMC.mu = randn(D_alpha,1); % the initial values for parameter \mu
                    initial_MCMC.sig2 = iwishrnd(eye(D_alpha),D_alpha + 10); % the initial values for \Sigma
                    initial_MCMC.a_d = 1./random('gam',1/2,1,D_alpha,1); 
                    [MCMC_initial, initial_MCMC] = VB_adaptive_Initialization(model{m},train_data,initial_MCMC,N_iter,R,epsilon);
                    initial = [reshape(MCMC_initial.alpha_store,D_alpha*J,1); MCMC_initial.mu_store; log(MCMC_initial.a_d_store)];
                    lambda.mu = initial;
                    lambda.B = zeros(p1,r)/r;    lambda.B = tril(lambda.B);
                    lambda.d = 0.01*ones(p1,1);
                    VB_settings.initial = lambda;
                    output = Hybrid_VAFC(model{m},train_data,@Likelihood_Hybrid,@prior_density_Hybrid,VB_settings);
                    convergence = output.converge;
                    count = count + 1;
                end
            else
                VB_settings.threshold = 1; % threshold for convergence criterion
                VB_settings.initial = lambda;
                output = Hybrid_VAFC(model{m},train_data,@Likelihood_Hybrid,@prior_density_Hybrid,VB_settings);
                count = 1;
                while output.converge == "no" && count <=5
                    VB_settings.threshold = 1 + (count>3)*(count-3); % threshold for convergence criterion
                    initial_MCMC.alpha = randn(D_alpha,J);
                    initial_MCMC.mu = randn(D_alpha,1); % the initial values for parameter \mu
                    initial_MCMC.sig2 = iwishrnd(eye(D_alpha),D_alpha + 10); % the initial values for \Sigma
                    initial_MCMC.a_d = 1./random('gam',1/2,1,D_alpha,1); 
                    [MCMC_initial, initial_MCMC] = VB_adaptive_Initialization(model{m},train_data,initial_MCMC,N_iter,R,epsilon);
                    initial = [reshape(MCMC_initial.alpha_store,D_alpha*J,1); MCMC_initial.mu_store; log(MCMC_initial.a_d_store)];
                    lambda.mu = initial;
                    lambda.B = zeros(p1,r)/r;    lambda.B = tril(lambda.B);
                    lambda.d = 0.01*ones(p1,1);
                    VB_settings.initial = lambda;
                    output = Hybrid_VAFC(model{m},train_data,@Likelihood_Hybrid,@prior_density_Hybrid,VB_settings);
                    convergence = output.converge;
                    count = count + 1;
                end                  
            end
            lambda = output.lambda;
            VB_results{k,m} = output;

     %===================== Compute the K-fold CVVB ELPD estimate =======================  

            if output.converge == "yes" % the threshold -5,000 for LB to stop when model is badly approximated, avoiding wasting time on this !
                N = 10000;   % number of draws to estimate the ELPD
                VB_samples = randn(N,p1+r)';
                z = VB_samples(1:r,:);
                eps = VB_samples(r+1:end,:);
                log_pdfs = zeros(N,1);
                mu = lambda.mu; B = lambda.B; d = lambda.d; 
                model_current = model{m}; 
                parfor i=1:N
                    theta_1 = mu + B*z(:,i) + d.*eps(:,i); 
                    ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
                    like = Likelihood_Hybrid(model_current,ALPHA,test_data,"no");
                    log_pdfs(i) = like.log;
                end
                max_log = max(log_pdfs);
                pdfs_ratio = exp(log_pdfs - max_log);
                ELPD_CVVB(k,m) = max_log + log(mean(pdfs_ratio));

                disp(['    Fold  ',num2str(k),' : || ELPD = ',num2str( round(ELPD_CVVB(k,m),1), '%0.1f'),...
                    ' || Initial LB = ',num2str(round(output.LB(1),1), '%0.1f'),...
                    ' || max LB = ',num2str( round(output.max_LB,1), '%0.1f'),...
                    ' || N_iter  = ',num2str(length(output.LB))]);
                k = k+1;
             %-----------  CHECK IT THE ALGORITHM CONVERGES --------------------
            else
                disp('    Warning: This model is badly approximated => skip !');
                k = K + 1;
            end

        end
        CPUtime = toc;
        disp(['    K-fold CVVB estimate of ELPD = ',num2str( round(mean(ELPD_CVVB(:,m)),2), '%0.2f' ),...
            ' & the running time is ',num2str(round(CPUtime/60,1), '%0.1f'),' minutes.']);
        save('CVVB_results.mat')
        disp('================================================================================================');
        % =======================================================================
        total_time = total_time + CPUtime;
    end

%% Step 4: Save the results
disp(['************************* The total running time is ',num2str(round(total_time/3600,2), '%0.1f'),' hours. *************************'])
save('CVVB_results.mat')