clear all
clc
%% DESCRIPTION: 
    
%% INPUT:

     
    
    exp_name = "Forstmann";
    data_name = 'Forstmann';
    model_index = [2 9 10 15 21 23 27];

    likelihood = str2func(strcat("likelihood_",exp_name,"_Hybrid")); % automatically choose the correct likelihood function
    VB_adaptive_Initialization = str2func(strcat("VB_", exp_name, "_adaptive_Initialization")); % automatically choose the correct initialization function


%% LOAD DATA and RANDOMLY SPLIT INTO K folds
load(['Data/' data_name '.mat'])
J = length(data.RT); % number of participants
K = 5; %number of folds
    train_ind = cell(K,J);  test_ind = cell(K,J);
    for j = 1:J
        n_j = length(data.RT{j});
        RandInd = randperm(n_j);
        fold_size = floor(n_j/K);
        for k = 1:K-1
            test_ind{k,j} = RandInd((k-1)*fold_size + 1 : k*fold_size);
            train_ind{k,j} = setdiff(RandInd,test_ind{k,j});
        end
        test_ind{K,j} = RandInd(k*fold_size + 1 : end);
        train_ind{K,j} = setdiff(RandInd,test_ind{K,j});
    end

%% get indices for all the models
if exp_name == "Mnemonic"
    M = 16; % total of competing models
    model = cell(1,M);
    name_c = ["R","E*R"];
    name_A = ["1","E"];
    name_v = ["S*M", "E*S*M"];
    name_s = ["M"];
    name_tau = name_A;
    num_ind = [1 2 4 8]; % number of random effects per subject
    m_ind = 1;
    for c_ind = 1:2
        for A_ind = 1:2
            for v_ind = 1:2
                for tau_ind = 1:2
                    model{m_ind}.name = [name_c(c_ind), name_A(A_ind), name_v(v_ind), name_s, name_tau(tau_ind)];
                    model{m_ind}.index = [num_ind(c_ind+1), num_ind(A_ind), num_ind(v_ind+2), 1, num_ind(tau_ind)];
                    m_ind = m_ind + 1;
                end
            end
        end
    end
elseif exp_name == "Lexical"
    M = 256; % total of competing models
    model = cell(1,M);
    name_b = ["1","C","E","C*E"];
    name_A = name_b;
    name_v = ["1","C","E","W","C*E","C*W","W*E","C*W*E"];
    name_tau = ["1","E"];
    num_ind = [1 2 2 4 4 8 8 16]; % number of random effects per subject
    m_ind = 1;
    for v_ind = 1:8
        for A_ind = 1:4
            for b_ind = 1:4
                for tau_ind = 1:2
                    model{m_ind}.name = [name_b(b_ind), name_A(A_ind), name_v(v_ind), "0", name_tau(tau_ind)];
                    model{m_ind}.index = [num_ind(b_ind), num_ind(A_ind), num_ind(v_ind), 0, num_ind(tau_ind)];
                    m_ind = m_ind + 1;
                end
            end
        end
    end
else
    M = 27; % total of competing models
    model = cell(1,M);
    
                        
    model_ind = [1 1 1; 1 1 2; 1 1 3; 1 2 3; 1 2 2; 1 2 1; 1 3 1; 1 3 2; 
        1 3 3;  2 3 3; 2 3 2; 2 3 1; 2 2 1; 2 2 2; 2 2 3; 2 1 3; 2 1 2; 2 1 1;
        3 1 1; 3 1 2; 3 1 3; 3 2 3; 3 2 2; 3 2 1; 3 3 1; 3 3 2; 3 3 3];% each row of matrix z is [z_c, z_v, z_tau]
    names = ["1", "2", "3"];
    for i = 1:M
        ind = model_ind(i,:);
        model{i}.name = [names(ind(1)), "1", names(ind(2)), "0", names(ind(3))];
        model{i}.index = zeros(5,1);
        model{i}.index(2) = 1;
        model{i}.index([1 3 5]) = ind;
        model{i}.index(3) = 2*model{i}.index(3);
    end

end
%% General VB approximation Setup

% Set up the VB approximation for the chosen model 
    log_scores = zeros(K+1,M);
    VB_results = cell(K+1,M); % Store all results of VB for later analysis if needed !
    
%% CVVB ALGORITHM
tic
for m = model_index
    
% ----------------- MODEL SETTING & SET UP PRIOR DISTRIBUTIONS -----------
    % theta = (alpha_1, ..., alpha_J, theta_G) with theta_G = (mu,Sigma,a_d)

    D_latent = sum(model{m}.index); % number of random effects per participant
    D_G = D_latent + D_latent*(D_latent+1)/2 + D_latent; % D_G = dim(theta_G)total number of global parameters
    p = D_latent*J + D_G; % dim(theta) = p;
    p1 = D_latent*(J + 2); % dim(theta_1) = p1;

                        % Prior setting
    prior_par.mu= zeros(D_latent,1); %the prior for \mu_{\alpha}
    prior_par.cov = eye(D_latent);%the prior for \Sigma_{\alpha}
    prior_par.v_a = 2; %the hyperparameters of the prior of \Sigma_{\alpha}
    prior_par.A_d = ones(D_latent,1); %the hyperparameters of the prior of \Sigma_{\alpha}
    model{m}.num_subjects = J;
%     model{m}.dim = p; % dim(theta) = p = number of all parameters
    
    model_name = ['c ~', model{m}.name(1),'| A ~',model{m}.name(2),'| v ~',model{m}.name(3),'| s ~',model{m}.name(4),'| t0 ~',model{m}.name(5)];
    disp([join(['Model ',num2str(m),' : ',model_name])]);
    
% ------------------ VB Tuning Parameter Setting -------------------------
                    % Set the MCMC tuning parameters
    max_iter = 10000; % number of iterations
    max_norm = 1000; % upper bound the the norm of the gradients

    r = 15 + sum(model{m}.index); % B has size dxr
    I = 10; % number of Monte Carlo samples to estimate the gradients
    window = 200;    patience_parameter = 200;

                            % Learning rates
    learning_rate.name = "ADADELTA";
    learning_rate.v = 0.95;     learning_rate.eps = 10^(-7);
                            
                            % Initialization
    if sum(model{m}.index) < 10 
        epsilon = 1; % scale parameter of covariance matrix
        N_iter = 100; % number of iterations in MCMC to initialize VB
    elseif sum(model{m}.index) < 20
        epsilon = 0.6;
        N_iter = 150; % number of iterations in MCMC to initialize VB
    else
        epsilon = 0.3;
        N_iter = 150; % number of iterations in MCMC to initialize VB
    end
    R = 10; % number of particles in PMwG
	initial_MCMC.alpha = randn(D_latent,J);
	initial_MCMC.mu = randn(D_latent,1); %the initial values for parameter \mu
	initial_MCMC.sig2 = iwishrnd(eye(D_latent),D_latent + 10); % the initial values for \Sigma
	initial_MCMC.a_d = 1./random('gam',1/2,1,D_latent,1);    

%=========================== CV VB FOR CHOSEN MODEL =======================

    %     for k = 1:K
    k = 1;  fold_k = 1; repetition = 1;
%         disp(['        Fold  1 : ']);
    while k<=(K+1)
        %(1) Split the dataset into train and test sets
        if k <(K+1)
            fold_k = k;
        else
            fold_k = 1;
        end
        if exp_name == "Lexical"
            for j=1:J
                train_data.RT{j} = data.RT{j}(train_ind{fold_k,j});  
                train_data.RE{j} = data.RE{j}(train_ind{fold_k,j});
                train_data.E{j} = data.E{j}(train_ind{fold_k,j});
                train_data.W{j} = data.W{j}(train_ind{fold_k,j});

                test_data.RT{j} = data.RT{j}(test_ind{fold_k,j});  
                test_data.RE{j} = data.RE{j}(test_ind{fold_k,j});
                test_data.E{j} = data.E{j}(test_ind{fold_k,j});
                test_data.W{j} = data.W{j}(test_ind{fold_k,j});
            end
        elseif exp_name == "Mnemonic"
            for j=1:J
                train_data.RT{j} = data.RT{j}(train_ind{fold_k,j});  
                train_data.RE{j} = data.RE{j}(train_ind{fold_k,j});
                train_data.E{j} = data.E{j}(train_ind{fold_k,j});
                train_data.M{j} = data.M{j}(train_ind{fold_k,j});
                train_data.S{j} = data.S{j}(train_ind{fold_k,j});

                test_data.RT{j} = data.RT{j}(test_ind{fold_k,j});  
                test_data.RE{j} = data.RE{j}(test_ind{fold_k,j});
                test_data.E{j} = data.E{j}(test_ind{fold_k,j});
                test_data.M{j} = data.M{j}(test_ind{fold_k,j});
                test_data.S{j} = data.S{j}(test_ind{fold_k,j});
            end
        else
            for j=1:J
                train_data.RT{j} = data.RT{j}(train_ind{fold_k,j});  
                train_data.RE{j} = data.RE{j}(train_ind{fold_k,j});
                train_data.E{j} = data.E{j}(train_ind{fold_k,j});
               
                test_data.RT{j} = data.RT{j}(test_ind{fold_k,j});  
                test_data.RE{j} = data.RE{j}(test_ind{fold_k,j});
                test_data.E{j} = data.E{j}(test_ind{fold_k,j});
               
            end   
        end
        threshold_good_initial = 3000; % LB threshold for not a bad initial value.
        if k==1 
            initial_LB = threshold_good_initial/2;
            count_initial = 0;
            while  count_initial <10
                count_initial =  count_initial + 1;
                [MCMC_initial, ~] = VB_adaptive_Initialization(model{m},train_data,initial_MCMC,N_iter,R,epsilon);
                initial = [reshape(MCMC_initial.alpha_store,D_latent*J,1); MCMC_initial.mu_store; log(MCMC_initial.a_d_store)];
                lambda.mu = initial;
                lambda.B = zeros(p1,r)/r;    lambda.B = tril(lambda.B);
                lambda.d = 0.01*ones(p1,1);            
                initial_LB = Hybrid_VAFC_LB(model{m},train_data,likelihood,@prior_density_Hybrid,prior_par,lambda,r,I);
                if (initial_LB < threshold_good_initial)
                    initial_MCMC.alpha = MCMC_initial.alpha_store;
                    initial_MCMC.mu = MCMC_initial.mu_store; %the initial values for parameter \mu
                    C_star = vech_inv(MCMC_initial.vech_C_star_store,D_latent);
                    C = C_star; C(1:D_latent+1:end) = exp(diag(C_star));
                    initial_MCMC.sig2 = C*C'; % the initial values for \Sigma
                    initial_MCMC.a_d = MCMC_initial.a_d_store;   
                    N_iter = 50; % 50 more iterations in MCMC !
                else
                    break
                end
            end
            disp(['          Initial LB = ' num2str(initial_LB) ' obtained by adpatively run MCMC ' num2str(count_initial) ' times!']);
        end
        output = Hybrid_VAFC(model{m},train_data,likelihood,@prior_density_Hybrid,prior_par,lambda,r,I,max_iter,window,patience_parameter,learning_rate,max_norm,"false");
        lambda = output.lambda;
        VB_results{k,m} = output;            
        if (std(output.LB_smooth(end-patience_parameter+1:end))<1)
                     %-------------------  ESTIMATE THE LOG-SCORE ------------------------
            N = 10000;   % number of draws to estimate the log score
            rqmc = normrnd_qmc(N,p1+r)';
            z = rqmc(1:r,:);
            eps = rqmc(r+1:end,:);
            log_pdfs = zeros(N,1);
            mu = lambda.mu; B = lambda.B; d = lambda.d; % to avoid parfor error
            model_current = model{m}; % to avoid parfor error
            parfor i=1:N
                theta_1 = mu + B*z(:,i) + d.*eps(:,i); %theta_vec = (alpha_1',...alpha_n',mu_alpha', log a') ';
                ALPHA = reshape(theta_1(1:D_latent*J),D_latent,J);
                like = feval(likelihood,model_current,ALPHA,test_data,"false");
                log_pdfs(i) = like.log;
            end
            max_log = max(log_pdfs);
            pdfs_ratio = exp(log_pdfs - max_log);
            log_scores(k,m) = max_log + log(mean(pdfs_ratio));

            disp(['    Fold  ',num2str(k),' : || log-score = ',num2str( log_scores(k,m)),' || Initial LB = ',num2str(output.LB(1)),' || first LB_smooth = ',num2str(output.LB_smooth(1)),' || last LB_smooth = ',num2str( output.LB_smooth(end)),' || max LB_smooth = ',num2str( output.max_LB),' || std of LB = ', num2str(std(output.LB_smooth(end-patience_parameter+1:end))),' || N_iter = ',num2str(length(output.LB))]);
%             save([save_dir save_name '.mat']); 
            k = k+1;
            repetition = 1;
        elseif (repetition <=5)
            disp(['                  Repetition ',num2str(repetition),' || Initial LB = ',num2str(output.LB(1)),' || first LB_smooth = ',num2str(output.LB_smooth(1)),' || last LB_smooth = ',num2str( output.LB_smooth(end)),' || max LB_smooth = ',num2str( output.max_LB),' || std of LB = ', num2str(std(output.LB_smooth(end-patience_parameter+1:end))),' || N_iter = ',num2str(length(output.LB))]);
            repetition =  repetition + 1;
        else
            k = K + 2;
            disp(['         This model is badly approximated => skip ! (LB max = ' num2str(output.max_LB) ')'])
        end       
    end
    disp(['Average log_score = ',num2str( mean(log_scores(2:K+1,m)))]);
    save([save_dir save_name '.mat']); 
    % =======================================================================
end
     
toc
CPUtime = toc;
%% SAVE RESULTS
save([save_dir save_name '.mat']); 
