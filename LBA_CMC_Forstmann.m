function [alpha,llh] = LBA_CMC_Forstmann(model,data,theta_G,alpha,...
J,D_alpha,R,mean_prop,cov_prop,better_proposal,epsilon)
%this is the Conditional Monte Carlo algorithm to sample the random effects
%for each subjects.
parfor j=1:J
    n_j = length(data.RT{j});    
% ------- (step 1): Generate alpha_j from the proposal distribution -------
%
% -------------------------------------------------------------------------   
    
    alpha_j_k = alpha(:,j); % the set of random effects from previous iteration of MCMC for conditioning.
    
    if better_proposal == true % use a better proposal
        % setting the weight of the mixture for the proposal in the sampling stage. 
        w_mix1 = 0.65; % w_mix1 = weight for estimated proposal  
        w_mix2 = 0.05; % w_mix2 = weight for prior
        w_mix3 = 1 - w_mix1 - w_mix2; % w_mix3 = weight for third component
        % generating the proposals from the mixture distribution in the sampling stage
        %-----------------------
        u = rand(R,1); 
        n1 = sum(u < w_mix1);                                                                     
        n2 = sum(u < w_mix2); 
        n3 = R-n1-n2;
        
        chol_sig2 = chol(theta_G.sig2,'lower');
        C_star = chol_sig2;     C_star(1:D_alpha+1:end) = log(diag(chol_sig2));
        vech_C_star = vech(C_star);
        
%       Denote:  x1 = alpha_j and x2 = theta_G = (mu vech_C*);
        x2 = [theta_G.mu; vech_C_star];  
        % mean_prop(:,j) = [alpha_j; mu; vech(C^*)]
        mu_1 = mean_prop(1:D_alpha,j); % mean of x1;
        mu_2 = mean_prop(D_alpha+1:end,j); % mean of x2
        
        S_11 = cov_prop(1:D_alpha,1:D_alpha,j);  % cov(x1,x1) 
        S_22 = cov_prop(D_alpha+1:end, D_alpha+1:end,j);  % cov(x2,x2)
        S_12 = cov_prop(1:D_alpha,D_alpha+1:end,j); % cov(x1,x2)   
        S_21 = S_12';
        M =  S_12/S_22;
        
        cond_mean = mu_1 + M*(x2-mu_2); % cond_mean = mean of x1|x2, this is the mean of the proposal in the sampling stage
        cond_var = S_11 - M*S_21; % computing the variance of the proposal in the sampling stage
        chol_cond_var = chol(cond_var,'lower');
        rnorm1 = cond_mean + chol_cond_var*randn(D_alpha,n1);
        rnorm3 = alpha_j_k + chol_cond_var*randn(D_alpha,n3);
        
        chol_covmat = chol(theta_G.sig2,'lower');
        rnorm2 = theta_G.mu + chol_covmat*randn(D_alpha,n2);
        
        alpha_j_R = [rnorm1 rnorm2 rnorm3]; % alpha_j_R = [alpha_j^1, ... alpha_j^R]. Particles alpha_j^r are stores in colums of matrix alpha_j_R
    %-----------------------
    else
    % generating the proposals from the mixture distribution in the burn in
    % and initial sampling stage
    %-----------------------
        w_mix = 0.5; % setting the weights of the mixture in the burn in and initial sampling stage.
        u = rand(R,1);
        id1 = (u<w_mix);
        n1 = sum(id1);
        n2 = R-n1;
        chol_covmat = chol(theta_G.sig2,'lower');
        rnorm1 = alpha_j_k + epsilon.*chol_covmat*randn(D_alpha,n1);
        rnorm2 = theta_G.mu + chol_covmat*randn(D_alpha,n2);

        alpha_j_R = [rnorm1 rnorm2];  % alpha_j_R = [alpha_j^1, ... alpha_j^R]. Particles alpha_j^r are stores in colums of matrix alpha_j_R
    
    %------------------------
    end   
    
    % set the first particles to the values of the random effects from the
    % previous iterations of MCMC for conditioning
    alpha_j_R(:,1) = alpha_j_k;
        
% -------------- (step 2): Compute the importance weights  ----------------
%
% -------------------------------------------------------------------------    
  
    % Duplicate data (y_j) and stack them into a column vector
    
    RT_j = repmat(data.RT{j},R,1);
    RE_j = repmat(data.RE{j},R,1); % RE = 1 (error) 2 (correct)
    
    % Compute the log p(y_j|alpha_j^r,theta_G)
    
%     Match obersvations to correct pairs of parameters. This is done manually. If change model, only need to change this.
    ind_acc = (data.E{j} == 1); ind_neutral = (data.E{j} == 2);   ind_speed = (data.E{j} == 3);       

        %     Below code is generalized for R alpha_j's
    alpha_j = cell(5,1);
    alpha_j{1} = exp(alpha_j_R(1:model.index(1),:))'; % C
    alpha_j{2} = exp(alpha_j_R(model.index(1)+1:sum(model.index(1:2)),:))'; % A
    alpha_j{3} = exp(alpha_j_R(sum(model.index(1:2))+1:sum(model.index(1:3)),:))'; % v
    alpha_j{4} = [exp(alpha_j_R(sum(model.index(1:3))+1:sum(model.index(1:4)),:))' ones(R,1)]; % S
    alpha_j{5} = exp(alpha_j_R(sum(model.index(1:4))+1:sum(model.index),:))'; % T0
    
    theta_j = cell(1,5);
    
            % Duplicate parameter c accordingly to model
    if (model.name(1)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        theta_j(1) = cellfun(@(x) repmat(reshape(M*x',n_j*R,1),1,2),alpha_j(1),'UniformOutput',false); 
    elseif (model.name(1)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        theta_j(1) = cellfun(@(x) repmat(reshape(M*x',n_j*R,1),1,2),alpha_j(1),'UniformOutput',false); 
    else
        theta_j(1) = cellfun(@(x) kron(x,ones(n_j,2)),alpha_j(1),'UniformOutput',false); 
    end    
        
            % Duplicate parameter A accordingly to model
            
    theta_j(2) = cellfun(@(x) kron(x,ones(n_j,2)),alpha_j(2),'UniformOutput',false); 
    
            % Duplicate parameter v accordingly to model
    if (model.name(3)== "3") 
        theta_j(3) = cellfun(@(x) kron(x(:,1:2),ind_acc) + kron(x(:,3:4),ind_neutral) + kron(x(:,5:6),ind_speed),alpha_j(3),'UniformOutput',false); 
    elseif (model.name(3)== "2") 
        theta_j(3) = cellfun(@(x) kron(x(:,1:2),(ind_acc + ind_neutral)) + kron(x(:,3:4),ind_speed),alpha_j(3),'UniformOutput',false); 
    else
        theta_j(3) = cellfun(@(x) kron(x,ones(n_j,1)),alpha_j(3),'UniformOutput',false); 
    end       
    
    
            % Duplicate parameter v accordingly to model
            
        theta_j{4} = ones(n_j*R,2);
        
        
            % Duplicate parameter tau accordingly to model
    if (model.name(5)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        theta_j(5) = cellfun(@(x) repmat(reshape(M*x',n_j*R,1),1,2),alpha_j(5),'UniformOutput',false); 
    elseif (model.name(5)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        theta_j(5) = cellfun(@(x) repmat(reshape(M*x',n_j*R,1),1,2),alpha_j(5),'UniformOutput',false); 
    else
        theta_j(5) = cellfun(@(x) kron(x,ones(n_j,2)),alpha_j(5),'UniformOutput',false); 
    end 
    
    theta_j{1} = theta_j{1} + theta_j{2}; % Convert theta_j{1} from C to B.
    

    FUNC = LBA_pdf(RE_j,RT_j,theta_j{1},theta_j{2},theta_j{3},theta_j{4},theta_j{5},"false");
    
    log_LBA_j = FUNC.log_element_wise;
    lw_reshape = reshape(log_LBA_j,n_j,R);
    logw_first = sum(lw_reshape)'; % 
    
    % Computing the log of p(\alpha|\theta) and density of the proposal for
    %burn in and initial sampling stage (count<=switch_num) and sampling
    %stage (count>switch_num)
    
    if  better_proposal == true
%         logw_second = logmvnpdf(alpha_j_R,theta_G.mu,theta_G.sig2);
        logw_second = log(mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2));
        logw_third = log(w_mix1.*mvnpdf(alpha_j_R',cond_mean',cond_var)+...
            w_mix2.*mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2)+ w_mix3.*mvnpdf(alpha_j_R',alpha_j_k',cond_var));
        logw = logw_first + logw_second - logw_third;
    else
%         logw_second = logmvnpdf(alpha_j_R,theta_G.mu,theta_G.sig2);
        logw_second = log(mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2));
        logw_third = log(w_mix.*mvnpdf(alpha_j_R',alpha_j_k',(epsilon^2).*theta_G.sig2)+...
            (1-w_mix).*mvnpdf(alpha_j_R',theta_G.mu',theta_G.sig2));
        logw = logw_first + logw_second - logw_third;
    end
    
    %check if there is imaginary number of logw
    
%     id = imag(logw)~=0;
%     id = 1-id;
%     id = logical(id);
%     logw = logw(id,1); 
%     logw = real(logw);
% 
%     if sum(isinf(logw))>0 | sum(isnan(logw))>0
%      id = isinf(logw) | isnan(logw);
%      id = 1-id;
%      id = logical(id);
%      logw = logw(id,1);
%     end
    
    max_logw = max(logw);
    weight = exp(logw-max_logw);
    llh_i(j) = max_logw+log(mean(weight)); 
%     llh_i(j) = real(llh_i(j)); 	
    weight = weight./sum(weight);
%     if sum(weight<0)>0
%         id = weight<0;
%         id = 1-id;
%         id = logical(id);
%         weight = weight(id,1);
%     end
    Nw = length(weight);
    
    if (sum(weight>0)>0) 
        ind = randsample(Nw,1,true,weight);
        alpha(:,j) = alpha_j_R(:,ind);
%     else
%         alpha(:,j) = alpha_j_k;
    end
%----------------------------------------------------------------------------------------------------------------------------------    
    
end
llh = sum(llh_i);
end

