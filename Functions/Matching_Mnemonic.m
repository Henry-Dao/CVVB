function [z_ij, z_j] = Matching_Mnemonic(model,alpha_j,data_subject_j)
%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        alpha_j = tranformed random effects (column vector/ matrix) of
%        subject j
%        data_subject_j = a structure containing data from subject j
% OUTPUT: z_ij = the natural random effects that match with the
% osbservations

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%          
    R = size(alpha_j,2);
    
    % ---------- Trasform random effects to the natural form ---------
    
    c_j = exp(alpha_j(1:model.dim(1),:))'; % C
    A_j = exp(alpha_j(model.dim(1)+1:sum(model.dim(1:2)),:))'; % A
    v_j = exp(alpha_j(sum(model.dim(1:2))+1:sum(model.dim(1:3)),:))'; % v
    s_j = exp(alpha_j(sum(model.dim(1:3))+1:sum(model.dim(1:4)),:))'; % S
    tau_j = exp(alpha_j(sum(model.dim(1:4))+1:sum(model.dim),:))'; % T0

    % ---------- Match the random effects with each observation ---------
    E_j = data_subject_j.E;
    S_j = data_subject_j.S;
    n_j = length(E_j);
    I_a = (E_j == "accuracy");     I_s = (E_j == "speed");   
    I_new = (S_j == "new");   I_old = (S_j == "old"); 
        
        % ----------- The threshold parameter c -----------------
        
    if (model.constraints(1) == "E*R") % c_j =  c^(s,o), c^(s,n), c^(a,o), c^(a,n) 
        c_ij = kron(c_j(:,1:2),I_s) + kron(c_j(:,3:4),I_a);  
    elseif (model.constraints(1) == "R") % c_j =  c^(o), c^(n)  
        c_ij = kron(c_j,ones(n_j,1));          
    end  
    
        % ----------- The start point parameter A -----------------   
        
    if (model.constraints(2) == "E")  
        A_ij = repmat(kron(A_j(:,1),I_s) + kron(A_j(:,2),I_a),1,2); 
    elseif (model.constraints(2) == "1")  
        A_ij = kron(A_j,ones(n_j,2));
    end
    
    % ----------- The drift rate mean v -----------------    
    
    if (model.constraints(3) == "S*M")  % v = v^(o,m), v^(o,mm), v^(n,m), v^(n,mm)
        v_ij_old = kron(v_j(:,1),I_old) + kron(v_j(:,4),I_new);
        v_ij_new = kron(v_j(:,2),I_old) + kron(v_j(:,3),I_new);
       
    elseif (model.constraints(3) == "E*S*M") 
%         v = [v^(s,o,m), v^(s,o,mm), v^(s,n,m), v^(s,n,mm),...
%              v^(a,o,m), v^(a,n,mm), v^(a,o,m), v^(a,o,mm)]
        I_so = I_s.*I_old; I_sn = I_s.*I_new;
        I_ao = I_a.*I_old; I_an = I_a.*I_new;
       
        v_ij_old = kron(v_j(:,1),I_so) + kron(v_j(:,5),I_ao) +...
            kron(v_j(:,4),I_sn) + kron(v_j(:,8),I_an);
        v_ij_new = kron(v_j(:,2),I_so) + kron(v_j(:,6),I_ao) +...
            kron(v_j(:,3),I_sn) + kron(v_j(:,7),I_an);      
    end
    v_ij = [v_ij_old v_ij_new];
    
    % ----------- The drift rate std s -----------------    
    
    s_ij_old = kron(s_j, I_old) + repmat(I_new,R,1);
    s_ij_new = kron(s_j, I_new) + repmat(I_old,R,1);
    s_ij = [s_ij_old s_ij_new];
    
    % ----------- The non-decision time parameter tau -----------------    
    if (model.constraints(5) == "E") 
        tau_ij = repmat(kron(tau_j(:,1),I_s) + kron(tau_j(:,2),I_a),1,2);
    elseif (model.constraints(5) == "1") 
        tau_ij = kron(tau_j,ones(n_j,2)); 
    end
    
    % ----------- threshold b -----------------
        b_ij = c_ij + A_ij; % threshold b_ij = c_ij + A_ij.     
    % ---------- Store the results in cell arrays ---------
     
        z_ij = cell(1,5);
        z_ij{1} = b_ij;   z_ij{2} = A_ij;   z_ij{3} = v_ij;
        z_ij{4} = s_ij;   z_ij{5} = tau_ij; 
                
        z_j = cell(5,1);
        z_j{1} = c_j;   z_j{2} = A_j;   z_j{3} = v_j;
        z_j{4} = s_j;   z_j{5} = tau_j;
         
end