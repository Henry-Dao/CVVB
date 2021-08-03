function grad_alpha_j = Matching_Gradients_Forstmann_model_3_1_1(model,LBA_pdf_j,z_j,data_subject_j)
%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        LBA_j = output of LBA_pdf function (structure)
%        z_j = output of Matching_function (structure)
%        data_subject_j = a structure contains all observations from
%        subject j
% OUTPUT: grad_alpha_j = the partial derivatives wrt the random effect
%        alpha_j

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%% Calculate the gradients with respect to \alpha_j
    grad_alpha_j = cell(1,5);
    E_j = data_subject_j.E;
    I_a = (E_j == 1); I_n = (E_j == 2);   I_s = (E_j == 3);
    
    %-------------- Evaluate the gradient of log_pdf wrt alpha_c ----------
    
    M = [I_a I_n I_s ];
    grad_alpha_j{1} = sum(sum(LBA_pdf_j.grad_b,2).*M,1) ; 
    grad_alpha_j{1} = grad_alpha_j{1}.*(z_j{1}-z_j{2});
    
    %-------------- Evaluate the gradient of log_pdf wrt alpha_A ----------
    
    grad_alpha_j{2} = sum(LBA_pdf_j.grad_A + LBA_pdf_j.grad_b,'all');  
    grad_alpha_j{2} = grad_alpha_j{2}.*(z_j{2}); 
    
    %-------------- Evaluate the gradient of log_pdf wrt alpha_v ----------
    
    grad_alpha_j{3} = sum(LBA_pdf_j.grad_v);
    grad_alpha_j{3} = grad_alpha_j{3}.*z_j{3};
    
    %-------------- Evaluate the gradient of log_pdf wrt alpha_s ----------
    
    grad_alpha_j{4} = [];   
    
    %-------------- Evaluate the gradient of log_pdf wrt alpha_tau --------
    
    grad_alpha_j{5} = sum(LBA_pdf_j.grad_tau,'all');
    grad_alpha_j{5} = grad_alpha_j{5}.*z_j{5};
                                                                            
end