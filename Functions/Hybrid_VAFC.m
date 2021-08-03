function output = Hybrid_VAFC(model,data,likelihood,prior_density,prior_par,lambda,r,I,max_iter,window,patience_parameter,learning_rate,max_norm,silent)
%% DESCRIPTION
% INPUT: model = structure
%        likelihood = this is input function, for the likelihood
%        prior_density = this is input function, for the prior
%        lambda = intial value ( structure with 3 fields: lambda.mu, lambda.B and lambda.d
%        p = dim(theta);    r = number of factors;  I = number of MC samples
%        learn_rate = "ADADELTA" or "ADAM" or 
%        max_iter = the maximum number of iterations
%        silent = "false" then print out iteration t and lower bounds LB
%% Initial Stage: run the first window iterations
    
    LB = zeros(max_iter,1);
    J = length(data.RT);
    D_alpha = sum(model.index);
    p = D_alpha*(J + 2);
    v_a = prior_par.v_a;
    df = v_a + D_alpha + J - 1;
    

        v = learning_rate.v; eps = learning_rate.eps; % hyperparameters
        E_g2 = zeros(p*(r+2),1); E_delta2 = zeros(p*(r+2),1);
    

    for t = 1:window
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;
        rqmc = normrnd_qmc(I,p+r);
        epsilon = rqmc(:,1:p)';
        z = rqmc(:,p+1:end)';
        parfor i = 1:I
        % (1) Generate theta  
            theta_1 = mu + B*z(:,i) + d.*epsilon(:,i); % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
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
            like = likelihood(model,ALPHA,data,"true"); % last argument = "true" means compute the gradient
            prior = prior_density(model,ALPHA,mu_alpha,a,Sigma_alpha,prior_par); %DO NOT worry about the red underline warning. Tested parfor vs for and gave the same results
            q_lambda = q_VAFC(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
%             grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad;
%             temp = grad_theta_1_LB + q_Sigma.grad*LBs(i);
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z(:,i)';
            gradd(:,i) = temp.*epsilon(:,i);   
        end
        % Estimate the Lower Bound
            LB(t) = mean(LBs); 
            if silent == "false" && (mod(t,100)==0)
                disp(['                 iteration ',num2str(t),' || LB: ',num2str(LB(t)), ' || standard error: ',num2str(std(LBs))]);
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
            lambda.B = tril(lambda.B); %IMPORTANT: Upper Triangle of B must be zero !!!
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
        rqmc = normrnd_qmc(I,p+r);
        epsilon = rqmc(:,1:p)';
        z = rqmc(:,p+1:end)';
        parfor i = 1:I
        % (1) Generate theta  
            theta_1 = mu + B*z(:,i) + d.*epsilon(:,i); % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
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
            like = likelihood(model,ALPHA,data,"true"); % last argument = "true" means compute the gradient
            prior = prior_density(model,ALPHA,mu_alpha,a,Sigma_alpha,prior_par); %DO NOT worry about the red underline warning. Tested parfor vs for and gave the same results
            q_lambda = q_VAFC(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;

        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
%             grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad;
%             temp = grad_theta_1_LB + q_Sigma.grad*LBs(i);
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z(:,i)';
            gradd(:,i) = temp.*epsilon(:,i);   
        end

        % Estimate the Lower Bound
        LB(t) = mean(LBs);    
        LB_smooth(t_smooth) = mean(LB(t-window+1:t));
        if (silent == "false") && (mod(t,100)==0)
            disp(['                 iteration ',num2str(t),'|| smooth LB: ',num2str(LB_smooth(t_smooth)), ' standard error: ',num2str(std(LBs))]);
        end
        % Stopping Rule:
        if (LB_smooth(t_smooth)< max_best) || (abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)
            patience = patience + 1;      
            
        else
            patience = 0;
            lambda_best = lambda;
            max_best = LB_smooth(t_smooth);
            if max_best < -10^(5)
                disp('Warning: Failed to converge, LB less than -10^5');
                converge = "false";
                break
            end
        end 
        if (patience>patience_parameter) % && (std(LB_smooth(t_smooth - patience_parameter+1:t_smooth)) < 1)
            stopping_rule = "true";
            converge = "true";
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
            lambda.B = tril(lambda.B); %IMPORTANT: Upper Triangle of B must be zero !!!
            lambda.d = vec_lambda((p*(r+1)+1):end); 
       t = t + 1;
       t_smooth = t_smooth + 1;
    end

%% OUTPUT
    output.lambda = lambda_best;
    output.LB = LB(1:t-1);
    output.LB_smooth = LB_smooth(1:t_smooth-1);
    output.max_LB = max_best;
    output.converge = converge; 
end
