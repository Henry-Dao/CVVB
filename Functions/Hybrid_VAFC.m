function output = Hybrid_VAFC(model,data,likelihood,prior_density,VB_settings)
%% DESCRIPTION: 
% This function is used to carefully choose an initial value for VB (by
% running PMwG with a small number of iterations (R = 50 to 100).

% INPUT: model = structure that contains model specifications
%        data = the data (in the correct format, see the user manual)
%        likelihood = likelihood function
%        prior_density = prior density
%        VB_setting = Settings of VB
%             initial = intial value for lambda( structure with 3 fields: lambda.mu, lambda.B and lambda.d)
%             r = number of factors used to parameterize the covariance matrix of q()
%             I = number of MC samples used to estimate the lower bound and the gradients
%             max_iter = the maximum number of iterations
%             window = window size to compute the average of the lower bounds
%             patience_parameter = the number of consecutive iterations in which the LB does not increase
%             max_norm = gradient clipping
%             learning_rate = ADADELTA learning rate parameter (v & eps)
%             silent = "no" (print out iteration t and the estimated lower bounds LB)

% OUTPUT: is a structure that contains several fields
%     output.lambda = the best lambda (corresponds to the highest lower bound);
%     output.LB = all the estimated lower bounds
%     output.LB_smooth = all the averaged lower bounds
%     output.max_LB = the highest lower bound
%     output.converge = "true" if the optimization converges and "false" if it fails to converge; 

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%% Initial Stage: run the first window iterations
    lambda = VB_settings.initial;
    I = VB_settings.I;
    r = VB_settings.r;
    max_iter = VB_settings.max_iter;
    max_norm = VB_settings.max_norm;
    window = VB_settings.window;
    patience_parameter = VB_settings.patience_parameter;   
    
    LB = zeros(max_iter,1);
    J = model.num_subjects;
    D_alpha = sum(model.dim);
    p = D_alpha*(J + 2);
    v_a = model.prior_par.v_a;
    df = v_a + D_alpha + J - 1;
    
    v = VB_settings.learning_rate.v; eps = VB_settings.learning_rate.eps; 
    E_g2 = zeros(p*(r+2),1); E_delta2 = zeros(p*(r+2),1);
    
    for t = 1:window
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;

        parfor i = 1:I
        % (1) Generate theta  
            epsilon = randn(p,1);
            z = randn(r,1);
            theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
            ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
            mu_alpha = theta_1(D_alpha*J + 1:D_alpha*(J+1));
            log_a = theta_1(D_alpha*(J+1)+1:end);
            a = exp(log_a);
            
            Psi = 2*v_a*diag(1./a);
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end            
            Sigma_alpha = iwishrnd(Psi,df);
        % (2) Calculate the likelihood, prior and q_vb          
            like = likelihood(model,ALPHA,data,"yes"); % last argument = "yes" means to compute the gradients 
            prior = prior_density(model,ALPHA,mu_alpha,Sigma_alpha,a); 
            q_lambda = q_VAFC(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z';
            gradd(:,i) = temp.*epsilon;   
        end
        % Estimate the Lower Bound
            LB(t) = mean(LBs); 
            if VB_settings.silent == "no"% && (mod(t,100)==0)
                disp(['                 iteration ',num2str(t),' || LB: ',num2str(round(LB(t),1), '%0.1f'),...
                    ' || standard error: ',num2str(round(std(LBs),2), '%0.2f')]);
            end

        % Estimate the gradients
            grad_mu = mean(gradmu,2);
            grad_B = mean(gradB,3);
            grad_D = mean(gradd,2);

            g = [grad_mu;grad_B(:);grad_D]; % Stack gradient of LB into 1 column 
        % Gradient clipping
            norm_g = norm(g);
            if norm_g > max_norm
                g = max_norm*g/norm_g;
            end
        % Update learning rate
           
            E_g2 = v*E_g2 + (1-v)*g.^2;
            rho = sqrt(E_delta2 + eps)./sqrt(E_g2+eps);
            Delta = rho.*g;
            E_delta2 = v*E_delta2 + (1-v)*Delta.^2;
            
        % Update Lambda
            vec_lambda = [lambda.mu;lambda.B(:);lambda.d];
            vec_lambda = vec_lambda + Delta;
            lambda.mu = vec_lambda(1:p);
            lambda.B = vec2mat(vec_lambda((p+1):(p*(r+1))),p)'; 
            lambda.B = tril(lambda.B); 
            lambda.d = vec_lambda((p*(r+1)+1):end);
    end

    %% Second stage: 
    
    LB_smooth = zeros(max_iter-window+1,1);
    LB_smooth(1) = mean(LB(1:t)); patience = 0; 
    lambda_best = lambda;
    max_best = LB_smooth(1);
    stopping_rule = "false";
    converge = "reach_max_iter";
    t = t + 1;
    t_smooth = 2; % iteration index for LB_smooth
    while t<= max_iter && stopping_rule == "false"        
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;
    
        parfor i = 1:I
        % (1) Generate theta  
            epsilon = randn(p,1);
            z = randn(r,1);
            theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
            ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
            log_a = theta_1(D_alpha*(J+1)+1:end);
            a = exp(log_a);
            mu_alpha = theta_1(D_alpha*J + 1:D_alpha*(J+1));
            Psi = 2*v_a*diag(1./a);
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end            
            Sigma_alpha = iwishrnd(Psi,df);
        % (2) Calculate the likelihood, prior and q_vb          
            like = likelihood(model,ALPHA,data,"yes"); 
            prior = prior_density(model,ALPHA,mu_alpha,Sigma_alpha,a); 
            q_lambda = q_VAFC(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;

        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z';
            gradd(:,i) = temp.*epsilon;   
        end

        % Estimate the Lower Bound
        LB(t) = mean(LBs);    
        LB_smooth(t_smooth) = mean(LB(t-window+1:t));
        if (VB_settings.silent == "no")
            disp(['                 iteration ',num2str(t),'|| smooth LB: ',num2str(round(LB_smooth(t_smooth),1), '%0.1f'),...
                ' standard error: ',num2str(round(std(LBs)/I,2), '%0.2f')]);
        end
        % Stopping Rule:
        if (LB_smooth(t_smooth)< max_best) || (abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)
            patience = patience + 1;                
        else
            patience = 0;
            lambda_best = lambda;
            max_best = LB_smooth(t_smooth);
        end 
        if (patience>patience_parameter) 
            stopping_rule = "true";
            if std(LB_smooth((t_smooth - patience_parameter + 1): t_smooth)) > 1
                disp(['    Warning: VB might not converge to a good local mode',...
                    '(Initial LB = ',num2str(round(LB(1),1), '%0.1f'),...
                    ' || max LB = ',num2str( round(max_best,1), '%0.1f'),')']);
                converge = "no";
            else
                converge = "yes";
            end
            
        end  
        
        % Estimate the gradients
            grad_mu = mean(gradmu,2);
            grad_B = mean(gradB,3);
            grad_D = mean(gradd,2);

            g = [grad_mu;grad_B(:);grad_D]; % Stack gradient of LB into 1 column 
        % Gradient clipping
            norm_g = norm(g);
            if norm_g > max_norm
                g = max_norm*g/norm_g;
            end
        % Update learning rate
            
            E_g2 = v*E_g2 + (1-v)*g.^2;
            rho = sqrt(E_delta2 + eps)./sqrt(E_g2+eps);
            Delta = rho.*g;
            E_delta2 = v*E_delta2 + (1-v)*Delta.^2;
            
        % Update Lambda  
            vec_lambda = [lambda.mu;lambda.B(:);lambda.d];
            vec_lambda = vec_lambda + Delta;
            lambda.mu = vec_lambda(1:p);
            lambda.B = vec2mat(vec_lambda((p+1):(p*(r+1))),p)'; 
            lambda.B = tril(lambda.B);
            lambda.d = vec_lambda((p*(r+1)+1):end); 
       t = t + 1;
       t_smooth = t_smooth + 1;
    end

%% Draws from VB distribution
if VB_settings.generated_samples == "yes"
    mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;   
    N = 10000;
    theta_VB = zeros(p + D_alpha*(D_alpha+1)/2,N);
    parfor i = 1:N
        epsilon = randn(p,1);
        z = randn(r,1);
        theta_1 = mu + B*z + d.*epsilon;
        ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
        mu_alpha = theta_1(D_alpha*J + 1:D_alpha*(J+1));
        log_a = theta_1(D_alpha*(J+1)+1:end);
        a = exp(log_a);            
        Psi = 2*v_a*diag(1./a);
        for j=1:J
            Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
        end            
        Sigma_alpha = iwishrnd(Psi,df);
        C = chol(Sigma_alpha,'lower');
        C_star = C; C_star(1:D_alpha+1:end) = log(diag(C));
        theta_VB(:,i) = [ALPHA(:); mu_alpha; vech(C_star); log_a];        
    end 
    output.theta_VB = theta_VB;
end
    
%% Save the output
    output.lambda = lambda_best;
    output.LB = LB(1:t-1);
    output.LB_smooth = LB_smooth(1:t_smooth-1);
    output.max_LB = max_best;
    output.converge = converge;    
end
