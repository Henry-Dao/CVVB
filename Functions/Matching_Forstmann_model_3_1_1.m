function [z_ij, z_j] = Matching_Forstmann_model_3_1_1(model,alpha_j,data_subject_j)
%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        alpha = tranformed random effects (column vector/ matrix)
%        data_subject_j = a structure contains all observations from
%        subject j
% OUTPUT: z_ij = the natural random effects that match with the
% osbservations

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
    R = size(alpha_j,2); 
    
    % ---------- Transform random effects to the natural form ---------
    
    c_j = exp(alpha_j(1:3,:))'; % seperate threshold c
    A_j = exp(alpha_j(4,:))'; % upper bound A
    v_j = exp(alpha_j(5:6,:))'; % drift rate means v
    s_j = ones(R,2); % drift rate standard devation s = 1
    tau_j = exp(alpha_j(7,:))'; % non-decision time tau

    % ---------- Match the random effects with each observation ---------   
    
    E_j = data_subject_j.E;
    n_j = length(E_j);
    
        % ----------- Seperate threshold c -----------------
        
        I_a = (E_j == 1); I_n = (E_j == 2);   I_s = (E_j == 3); 
        M = [I_a I_n I_s ];
        c_ij = repmat(reshape(M*c_j',n_j*R,1),1,2);  

        % ----------- Upper bound A -----------------
        
        A_ij = kron(A_j,ones(n_j,2)); 
        
        % ----------- non-decision time -----------------    
        
        tau_ij = kron(tau_j,ones(n_j,2)); 
        
        % ----------- drift rate means -----------------
        
        v_ij = kron(v_j,ones(n_j,1));       

        % ----------- drift rate std -----------------

        s_ij = ones(n_j*R,2);

        % ----------- threshold b -----------------
        
        b_ij = c_ij + A_ij; % threshold b_ij = c_ij + A_ij.
        b_j = c_j + A_j; % threshold b_j = c_j + A_j.
        
        % ---------- Store the results in cell arrays ---------
     
        z_ij = cell(5,1);
        z_ij{1} = b_ij;   z_ij{2} = A_ij;   z_ij{3} = v_ij;
        z_ij{4} = s_ij;   z_ij{5} = tau_ij; 
                
        z_j = cell(5,1);
        z_j{1} = b_j;   z_j{2} = A_j;   z_j{3} = v_j;
        z_j{4} = s_j;   z_j{5} = tau_j;
        
end