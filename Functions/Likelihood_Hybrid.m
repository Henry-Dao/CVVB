function output = Likelihood_Hybrid(model,alpha,data,gradient)
%% DESCRIPTION: 
% This is the likelihood function in Hyrbid VB for Hierarchical LBA models
% INPUT: model = structure that contains model specifications
%        alpha = tranformed random effects (matrix)
%        data = numerical vector that indicates the experimental conditions
%        gradient = evaluate the gradient of log likelihood if gradient = "yes"
% OUTPUT: is a structure that contains several fields
%     output.log = log of the likelihood function;
%     output.grad = gradients of log likelihood
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%
    loga = 0; % log of the total likelihood

    D_alpha = sum(model.dim); % n_alpha = D_alpha = dimension of random effect alpha
    J = model.num_subjects; % number of subjects/participants
    grad_alpha = zeros(D_alpha,J); % store all the gradients wrt alpha_j
    grad_mu = zeros(D_alpha,1);
    grad_loga = zeros(D_alpha,1);
    Matching_Function_1 = str2func(model.matching_function_1);
    Matching_Function_2 = str2func(model.matching_function_2);
    for j = 1:J        
    %% Match each observation to the correct set of parameters (b,A,v,s,tau)
    
        [z_ij, z_j] = Matching_Function_1(model,alpha(:,j),data{j});
        
    %% Compute the gradients
        LBA_j = LBA_pdf(data{j}.RE,data{j}.RT,z_ij{1},z_ij{2},z_ij{3},...
            z_ij{4},z_ij{5},gradient);
        loga = loga + LBA_j.log;
        
    %% Match the gradients 
        if gradient == "yes"
            grad_alpha_j = Matching_Function_2(model,LBA_j,z_j,data{j});    
            grad_alpha(:,j) = [grad_alpha_j{1}'; grad_alpha_j{2}'; grad_alpha_j{3}';...
                grad_alpha_j{4}'; grad_alpha_j{5}'];
        end        
    end   
%% output of the function
    output.log = loga;
    if gradient ==  "yes"
        output.grad = [grad_alpha(:); grad_mu; grad_loga];
    end
end