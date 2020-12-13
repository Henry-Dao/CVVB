function [alpha,llh] = LBA_CMC_Mnemonic(model,data,theta_G,alpha,...
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
        
% -------------- (step 2): STACK RANDOM EFFECTS & RT ----------------
%
% -------------------------------------------------------------------------    
  
    % Duplicate data (y_j) and stack them into a column vector
    
    RT_j = repmat(data.RT{j},R,1);
    
    % Compute the log p(y_j|alpha_j^r,theta_G)
    
%     Match obersvations to correct pairs of parameters. This is done manually. If change model, only need to change this.
    ind_acc = (data.E{j} == "accuracy");     ind_speed = (data.E{j} == "speed");   
    ind_S_new = (data.S{j} == "new");   ind_S_old = (data.S{j} == "old"); 
    ind_R_new = (data.S{j} == "new");   ind_R_old = (data.S{j} == "old");
    ind_match = (data.M{j} == "TRUE");  ind_mismatch = (data.M{j} == "FALSE");
    
    ind_Snew_match = ind_S_new.*ind_match;   ind_Snew_mismatch = ind_S_new.*ind_mismatch;
    ind_Sold_match = ind_S_old.*ind_match;   ind_Sold_mismatch = ind_S_old.*ind_mismatch;

        %     Below code is generalized for R alpha_j's
        
    c_j = exp(alpha_j_R(1:model.index(1),:))'; % C
    A_j = exp(alpha_j_R(model.index(1)+1:sum(model.index(1:2)),:))'; % A
    v_j = exp(alpha_j_R(sum(model.index(1:2))+1:sum(model.index(1:3)),:))'; % v
    s_j = exp(alpha_j_R(sum(model.index(1:3))+1:sum(model.index(1:4)),:))'; % S
    tau_j = exp(alpha_j_R(sum(model.index(1:4))+1:sum(model.index),:))'; % T0
    
        % ----------- The threshold parameter c -----------------
    if (model.name(1) == "R") % c_j =  c^(n), c^(o) 
        c_j_stack_f = kron(c_j(:,1),ind_R_new) +  kron(c_j(:,2),ind_R_old);
        c_j_stack_F = kron(c_j(:,2),ind_R_new) +  kron(c_j(:,1),ind_R_old);

    else % c_j =  c^(n,s), c^(o,s), c^(n,a), c^(o,a)
        ind_Rnew_speed = ind_R_new.*ind_speed;   ind_Rold_speed = ind_R_old.*ind_speed;
        ind_Rnew_acc = ind_R_new.*ind_acc;   ind_Rold_acc = ind_R_old.*ind_acc;  
        
        c_j_stack_f = kron(c_j(:,1),ind_Rnew_speed) + kron(c_j(:,2),ind_Rold_speed) + ...
            kron(c_j(:,3),ind_Rnew_acc) + kron(c_j(:,4),ind_Rold_acc);
        c_j_stack_F = kron(c_j(:,2),ind_Rnew_speed) + kron(c_j(:,1),ind_Rold_speed) + ...
            kron(c_j(:,4),ind_Rnew_acc) + kron(c_j(:,3),ind_Rold_acc);       
    end
    
    % ----------- The start point parameter A -----------------    
    if (model.name(2) == "1")  % A = A
        A_j_stack = kron(A_j,ones(n_j,1)); 
    else % A = A^(s) A^(a)
        A_j_stack = kron(A_j(:,1),ind_speed) + kron(A_j(:,2),ind_acc);
    end
    
    % ----------- The drift rate mean v -----------------    
    
    if (model.name(3) == "S*M")  % v = v^(n,m), v^(n,mm), v^(o,m), v^(o,mm)             
        v_j_stack_f = kron(v_j(:,1),ind_Snew_match) + kron(v_j(:,2),ind_Snew_mismatch) + ...
            kron(v_j(:,3),ind_Sold_match) + kron(v_j(:,4),ind_Sold_mismatch); 
        v_j_stack_F = kron(v_j(:,2),ind_Snew_match) + kron(v_j(:,1),ind_Snew_mismatch) + ...
            kron(v_j(:,4),ind_Sold_match) + kron(v_j(:,3),ind_Sold_mismatch);  
    else % v = v^(s,n,m), v^(s,n,mm), v^(s,o,m), v^(s,o,mm), v^(a,n,m), v^(a,n,mm), v^(a,o,m), v^(a,o,mm)
        ind_nms = ind_speed.*ind_Snew_match; ind_oms = ind_speed.*ind_Sold_match;
        ind_nmms = ind_speed.*ind_Snew_mismatch; ind_omms = ind_speed.*ind_Sold_mismatch;
        
        ind_nma = ind_acc.*ind_Snew_match; ind_oma = ind_acc.*ind_Sold_match;
        ind_nmma = ind_acc.*ind_Snew_mismatch; ind_omma = ind_acc.*ind_Sold_mismatch;
        v_j_stack_f = kron(v_j(:,1),ind_nms) + kron(v_j(:,2),ind_nmms) +...
                      kron(v_j(:,3),ind_oms) + kron(v_j(:,4),ind_omms) +...
                      kron(v_j(:,5),ind_nma) + kron(v_j(:,6),ind_nmma) +...
                      kron(v_j(:,7),ind_oma) + kron(v_j(:,8),ind_omma);
                  
        v_j_stack_F = kron(v_j(:,2),ind_nms) + kron(v_j(:,1),ind_nmms) +...
                      kron(v_j(:,4),ind_oms) + kron(v_j(:,3),ind_omms) +...
                      kron(v_j(:,6),ind_nma) + kron(v_j(:,5),ind_nmma) +...
                      kron(v_j(:,8),ind_oma) + kron(v_j(:,7),ind_omma);     
    end
    
    % ----------- The drift rate std s -----------------    
    % s_j = s^(mm) since s^(m) = 1
    s_j_stack_f = kron(ones(R,1),ind_match) + kron(s_j,ind_mismatch);
    s_j_stack_F = kron(s_j,ind_match) + kron(ones(R,1),ind_mismatch);
    
    % ----------- The non-decision time parameter tau -----------------    
    if (model.name(5) == "1")  % tau = tau
        tau_j_stack = kron(tau_j,ones(n_j,1)); 
    else % tau = tau^(s) tau^(a)
        tau_j_stack = kron(tau_j(:,1),ind_speed) + kron(tau_j(:,2),ind_acc);
    end
    
    b_j_stack_f = c_j_stack_f + A_j_stack; % Convert from C to B.
    b_j_stack_F = c_j_stack_F + A_j_stack; % Convert from C to B.
    
% -------------- (step 3): COMPUTE THE LOG LBA ----------------
%
% -------------------------------------------------------------------------    
    f_c = pdf_c(RT_j,b_j_stack_f,A_j_stack,v_j_stack_f,s_j_stack_f,tau_j_stack,"false");

    F_k = CDF_c(RT_j,b_j_stack_F,A_j_stack,v_j_stack_F,s_j_stack_F,tau_j_stack,"false");

    %Step 4: Compute the log of LBA    
    log_LBA_j = log(f_c.value) + log(F_k.substract);
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
    
%     %check if there is imaginary number of logw (from David)
%     
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

