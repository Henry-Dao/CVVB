function [z_ij, z_j] = Matching_Lexical(model,alpha_j,data_subject_j)
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
    c_j = exp(alpha_j(1:model.dim(1),:))'; % seperate threshold c
    A_j = exp(alpha_j(model.dim(1)+1:sum(model.dim(1:2)),:))'; % upper bound A
    v_j = exp(alpha_j(sum(model.dim(1:2))+1:sum(model.dim(1:3)),:))'; % drift rate means v
    s_j = [exp(alpha_j(sum(model.dim(1:3))+1:sum(model.dim(1:4)),:))' ones(R,1)]; % drift rate std s
    tau_j = exp(alpha_j(sum(model.dim(1:4))+1:sum(model.dim),:))'; % non-decision time \tau
    
    % ----------- Match the random effects with each observation ----------
    E_j = data_subject_j.E;
    W_j = data_subject_j.W;
    n_j = length(E_j);
    I_s = (E_j == 1);  I_a = (E_j == 2);  
    I_hf = (W_j == 1);  I_lf = (W_j == 2);  I_vlf = (W_j == 3);  
    I_nw = (W_j == 4);
    
        % ------------------ The threshold parameter c --------------------
        
    if (model.constraints(1) == "C*E") % c_j = c^(s,e) c^(s,c) c^(a,e) c^(a,c)
        c_ij = kron(c_j(:,1:2),I_s) + kron(c_j(:,3:4),I_a);  
    elseif (model.constraints(1) == "E") % c_j = c^(s) c^(a)   
        M = [I_s I_a];
        c_ij = repmat(reshape(M*c_j',n_j*R,1),1,2);  
    elseif (model.constraints(1) == "C") 
        c_ij = kron(c_j,ones(n_j,1));
    elseif (model.constraints(1) == "1") % c_j = c  
        c_ij = kron(c_j,ones(n_j,2));
    end  

        % ------------------ The start point parameter A ------------------   
        
    if (model.constraints(2) == "C*E") % A_j = A^(s,e) A^(s,c) A^(a,e) A^(a,c)
        A_ij = kron(A_j(:,1:2),I_s) + kron(A_j(:,3:4),I_a);  
    elseif (model.constraints(2) == "E") % A_j = A^(s) A^(a)    
        M = [I_s I_a];
        A_ij = repmat(reshape(M*A_j',n_j*R,1),1,2);  
    elseif (model.constraints(2) == "C") % A_j = A^(e) A^(c)
        A_ij = kron(A_j,ones(n_j,1));
    elseif (model.constraints(2) == "1") % A_j = A
        A_ij = kron(A_j,ones(n_j,2));
    end     

        % -------------------- The drift rate mean v ----------------------   
        
    if (model.constraints(3) == "C*W*E")   
        I_hf_s = I_hf.*I_s;   I_lf_s = I_lf.*I_s;   I_vlf_s = I_vlf.*I_s; I_nw_s = I_nw.*I_s;
        I_hf_a = I_hf.*I_a;   I_lf_a = I_lf.*I_a;   I_vlf_a = I_vlf.*I_a; I_nw_a = I_nw.*I_a;
        v_ij = kron(v_j(:,1:2),I_hf_s) + kron(v_j(:,3:4),I_lf_s) + ...
            kron(v_j(:,5:6),I_vlf_s) + kron(v_j(:,7:8),I_nw_s) + ...
            kron(v_j(:,9:10),I_hf_a) + kron(v_j(:,11:12),I_lf_a) +...
            kron(v_j(:,13:14),I_vlf_a) + kron(v_j(:,15:16),I_nw_a);   
    elseif (model.constraints(3) == "C*W")   
        v_ij = kron(v_j(:,1:2),I_hf) + kron(v_j(:,3:4),I_lf) + ... 
               kron(v_j(:,5:6),I_vlf) + kron(v_j(:,7:8),I_nw);  
    elseif (model.constraints(3) == "C*E")  
        v_ij = kron(v_j(:,1:2),I_s) + kron(v_j(:,3:4),I_a);  
    elseif (model.constraints(3) == "W*E")   
        temp1 = [I_hf I_lf I_vlf I_nw].*I_s;
        temp2 = [I_hf I_lf I_vlf I_nw].*I_a;
        M = [temp1 temp2];
        v_ij = repmat(reshape(M*v_j',n_j*R,1),1,2);
     elseif (model.constraints(3) == "W")   
        M = [I_hf I_lf I_vlf I_nw];
        v_ij = repmat(reshape(M*v_j',n_j*R,1),1,2); 
    elseif (model.constraints(3) == "E")   
        M = [I_s I_a];
        v_ij = repmat(reshape(M*v_j',n_j*R,1),1,2); 
    elseif (model.constraints(3) == "C") 
        v_ij = kron(v_j,ones(n_j,1));
    elseif (model.constraints(3) == "1")  
        v_ij = kron(v_j,ones(n_j,2));
    end       
 
        % -------------------- The drift rate std s -----------------------    
     
    s_ij = ones(R*n_j,2);
    
        % ----------- The non-decision time parameter tau -----------------    
    if (model.constraints(5) == "E") 
        M = [I_s I_a];
        tau_ij = repmat(reshape(M*tau_j',n_j*R,1),1,2); 
    elseif (model.constraints(5) == "1") 
        tau_ij = kron(tau_j,ones(n_j,2)); 
    end        
    
    % ------------------------- threshold b ------------------------------
    
        b_ij = c_ij + A_ij; % threshold b_ij = c_ij + A_ij.  
        
    % ----------------- Store the results in cell arrays ------------------
    
        z_ij = cell(1,5);
        z_ij{1} = b_ij;   z_ij{2} = A_ij;   z_ij{3} = v_ij;
        z_ij{4} = s_ij;   z_ij{5} = tau_ij; 
                
        z_j = cell(5,1);
        z_j{1} = c_j;   z_j{2} = A_j;   z_j{3} = v_j;
        z_j{4} = s_j;   z_j{5} = tau_j;
         
end