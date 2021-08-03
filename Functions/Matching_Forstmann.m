function [z_ij, z_j] = Matching_Forstmann(model,alpha_j,data_subject_j)
%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        alpha_j = tranformed random effects of subject j(column vector/ matrix)
%        data_subject_j = a structure contains all observations from
%        subject j
%  ( E = 1(accuracy emphasis), E = 2(neutral emphasis) and E = 3(speed emphasis) )
% OUTPUT: z_ij = the natural random effects that match with the
% osbservations

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%          
    R = size(alpha_j,2);
    E_j = data_subject_j.E;
    % ---------- Trasform random effects to the natural form ---------    
    c_j = exp(alpha_j(1:model.dim(1),:))'; 
    A_j = exp(alpha_j(model.dim(1)+1:sum(model.dim(1:2)),:))'; 
    v_j = exp(alpha_j(sum(model.dim(1:2))+1:sum(model.dim(1:3)),:))'; 
    s_j = [exp(alpha_j(sum(model.dim(1:3))+1:sum(model.dim(1:4)),:))' ones(R,1)]; 
    tau_j = exp(alpha_j(sum(model.dim(1:4))+1:sum(model.dim),:))'; 

    % ---------- Match the random effects with each observation ---------
    I_a = ( E_j == 1); I_n = (E_j == 2);   I_s = (E_j == 3); 
    n_j = length(E_j);
    
        % ----------- Seperate threshold c -----------------
        
        if (model.constraints(1)== "3") 
            M = [I_a I_n I_s ];
            c_ij = repmat(reshape(M*c_j',n_j*R,1),1,2); 
        elseif (model.constraints(1)== "2") 
            M = [(I_a + I_n) I_s ];
            c_ij = repmat(reshape(M*c_j',n_j*R,1),1,2); 
        else
            c_ij = kron(c_j,ones(n_j,2)); 
        end    

        % ----------- Upper bound A -----------------
        
        A_ij = kron(A_j,ones(n_j,2)); 

        % ----------- drift rate means -----------------
        
        if (model.constraints(3)== "3") 
            v_ij = kron(v_j(:,1:2),I_a) + kron(v_j(:,3:4),I_n) + ...
                kron(v_j(:,5:6),I_s); 
        elseif (model.constraints(3)== "2") 
            v_ij = kron(v_j(:,1:2),(I_a + I_n)) + kron(v_j(:,3:4),I_s); 
        else
            v_ij = kron(v_j,ones(n_j,1)); 
        end       

        % ----------- drift rate std -----------------

            s_ij = ones(n_j*R,2);
            
        % ----------- non-decision time -----------------       
        if (model.constraints(5)== "3") 
            M = [I_a I_n I_s ];
            tau_ij = repmat(reshape(M*tau_j',n_j*R,1),1,2); 
        elseif (model.constraints(5)== "2") 
            M = [(I_a + I_n) I_s ];
            tau_ij = repmat(reshape(M*tau_j',n_j*R,1),1,2); 
        else
            tau_ij = kron(tau_j,ones(n_j,2)); 
        end 
        
        % ----------- threshold b -----------------
        b_ij = c_ij + A_ij; % threshold b_ij = c_ij + A_ij.
        b_j = c_j + A_j; % threshold b_j = c_j + A_j.
        
     % ---------- Store the results in cell arrays ---------
     
        z_ij = cell(1,5);
        z_ij{1} = b_ij;   z_ij{2} = A_ij;   z_ij{3} = v_ij;
        z_ij{4} = s_ij;   z_ij{5} = tau_ij; 
                
        z_j = cell(5,1);
        z_j{1} = b_j;   z_j{2} = A_j;   z_j{3} = v_j;
        z_j{4} = s_j;   z_j{5} = tau_j;

end