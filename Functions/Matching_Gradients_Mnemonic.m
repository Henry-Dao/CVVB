function grad_alpha_j = Matching_Gradients_Mnemonic(model,LBA_pdf_j,z_j,data_subject_j)
%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        LBA_j = output of LBA_pdf_j function (structure)
%        z_j = output of Matching_function (structure)
%        data_subject_j = a structure contains all observations from
%        subject j
% OUTPUT: grad_alpha_j = the partial derivatives wrt the random effect
%        alpha_j

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%% Calculate the gradients with respect to \alpha_j
    grad_alpha_j = cell(5,1);
    E_j = data_subject_j.E;
    S_j = data_subject_j.S;
    I_a = (E_j == "accuracy");     I_s = (E_j == "speed");   
    I_new = (S_j == "new");   I_old = (S_j == "old"); 
    
%-------------- Evaluate the gradient of log_pdf of alpha_c ---------------

    if (model.constraints(1) == "E*R") % c_j =  c^(s,o), c^(s,n), c^(a,o), c^(a,n) 
        grad_alpha_j{1} = sum([LBA_pdf_j.grad_b.*I_s,...
                               LBA_pdf_j.grad_b.*I_a],1);  
    elseif (model.constraints(1) == "R") % c_j =  c^(o), c^(n)  
        grad_alpha_j{1} = sum(LBA_pdf_j.grad_b,1);          
    end    
    grad_alpha_j{1} = grad_alpha_j{1} .*z_j{1};
    
%-------------- Evaluate the gradient of log_pdf of alpha_A ---------------

    if (model.constraints(2) == "E") 
        M = [I_s I_a];
        grad_alpha_j{2} = sum(sum(LBA_pdf_j.grad_A +...
                                  LBA_pdf_j.grad_b,2).*M,1);
    elseif (model.constraints(2) == "1")  
        grad_alpha_j{2} = sum(LBA_pdf_j.grad_A + LBA_pdf_j.grad_b,'all');
    end
    grad_alpha_j{2} = grad_alpha_j{2}.*z_j{2}; 
    
%-------------- Evaluate the gradient of log_pdf of alpha_v ---------------

    if (model.constraints(3) == "S*M")  % v = v^(o,m), v^(o,mm), v^(n,m), v^(n,mm)
        grad_alpha_j{3} = sum([LBA_pdf_j.grad_v.*I_old,...
                  [LBA_pdf_j.grad_v(:,2) LBA_pdf_j.grad_v(:,1)].*I_new],1);       
    elseif (model.constraints(3) == "E*S*M") 
%         v = [v^(s,o,m), v^(s,o,mm), v^(s,n,m), v^(s,n,mm),...
%              v^(a,o,m), v^(a,n,mm), v^(a,o,m), v^(a,o,mm)]
        I_so = I_s.*I_old; I_sn = I_s.*I_new;
        I_ao = I_a.*I_old; I_an = I_a.*I_new;       
        grad_alpha_j{3} = sum([LBA_pdf_j.grad_v.*I_so,...
                  [LBA_pdf_j.grad_v(:,2) LBA_pdf_j.grad_v(:,1) ].*I_sn,...
                   LBA_pdf_j.grad_v.*I_ao,...
                  [LBA_pdf_j.grad_v(:,2) LBA_pdf_j.grad_v(:,1) ].*I_an],1);    
    end  
    grad_alpha_j{3} = grad_alpha_j{3}.*z_j{3};
    
%-------------- Evaluate the gradient of log_pdf wrt alpha_s --------------

    grad_alpha_j{4} = sum(LBA_pdf_j.grad_s(:,1).*I_old +...
                          LBA_pdf_j.grad_s(:,2).*I_new,1);
    grad_alpha_j{4} = grad_alpha_j{4}.*z_j{4};
    
%-------------- Evaluate the gradient of log_pdf of alpha_tau -------------

    if (model.constraints(5) == "E") 
        M = [I_s I_a];
        grad_alpha_j{5} = sum(sum(LBA_pdf_j.grad_tau,2).*M,1);
    elseif (model.constraints(5) == "1") 
        grad_alpha_j{5} = sum(LBA_pdf_j.grad_tau,'all');
    end
    grad_alpha_j{5} = grad_alpha_j{5}.*z_j{5};
end