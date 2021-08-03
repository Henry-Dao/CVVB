function grad_alpha_j = Matching_Gradients_Forstmann(model,LBA_pdf,z_j,data_subject_j)
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
    ind_acc = (E_j == 1); ind_neutral = (E_j == 2);   ind_speed = (E_j == 3);
    
%-------------- Evaluate the gradient of log_pdf of alpha_c ------------
    if (model.constraints(1)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        grad_alpha_j{1} = sum(sum(LBA_pdf.grad_b,2).*M,1); 
    elseif (model.constraints(1)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        grad_alpha_j{1} = sum(sum(LBA_pdf.grad_b,2).*M,1);
    else
        grad_alpha_j{1} = sum(LBA_pdf.grad_b,'all');
    end
    grad_alpha_j{1} = grad_alpha_j{1}.*(z_j{1}-z_j{2});
    
%-------------- Evaluate the gradient of log_pdf of alpha_A ------------
    grad_alpha_j{2} = sum(LBA_pdf.grad_A + LBA_pdf.grad_b,'all').*(z_j{2}); 
    
%-------------- Evaluate the gradient of log_pdf of alpha_v ------------
    if (model.constraints(3)== "3")
        grad_alpha_j{3} = [sum(LBA_pdf.grad_v.*ind_acc) ...
            sum(LBA_pdf.grad_v.*ind_neutral) sum(LBA_pdf.grad_v.*ind_speed)]; 
    elseif (model.constraints(3)== "2") 
        grad_alpha_j{3} = [sum(LBA_pdf.grad_v.*(ind_acc + ind_neutral)) ...
            sum(LBA_pdf.grad_v.*ind_speed)];
    else
        grad_alpha_j{3} = sum(LBA_pdf.grad_v);
    end 
    grad_alpha_j{3} = grad_alpha_j{3}.*z_j{3};
%-------------- Evaluate the gradient of log_pdf wrt alpha_s ------------
    grad_alpha_j{4} = []; 
    
%-------------- Evaluate the gradient of log_pdf of alpha_tau ------------
    if (model.constraints(5)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        grad_alpha_j{5} = sum(sum(LBA_pdf.grad_tau,2).*M,1);
    elseif (model.constraints(5)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        grad_alpha_j{5} = sum(sum(LBA_pdf.grad_tau,2).*M,1);
    else
        grad_alpha_j{5} = sum(LBA_pdf.grad_tau,'all');
    end
    grad_alpha_j{5} = grad_alpha_j{5}.*z_j{5};
end