function LB = Hybrid_VAFC_LB(model,data,likelihood,prior_density,prior_par,lambda,r,I)
%% DESCRIPTION
% This functions is modified based on the function 'Hybrid_VAFC.m'
% Objective is to estimate the LB !
%% Initial Stage: run the first window iterations
    
    J = length(data.RT);
    D_alpha = sum(model.index);
    p = D_alpha*(J + 2);
    v_a = prior_par.v_a;
    df = v_a + D_alpha + J - 1;
 
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
        like = likelihood(model,ALPHA,data,"false"); % last argument = "true" means compute the gradient
        prior = prior_density(model,ALPHA,mu_alpha,a,Sigma_alpha,prior_par); %DO NOT worry about the red underline warning. Tested parfor vs for and gave the same results
        q_lambda = q_VAFC(theta_1,mu,B,d); 
        q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
        LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
 
    end
        % Estimate the Lower Bound
    LB = mean(LBs); 
    
end
