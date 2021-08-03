function grad_alpha_j = Matching_Gradients_Lexical(model,LBA_pdf_j,z_j,data_subject_j)
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
    E_j = data_subject_j.E;
    W_j = data_subject_j.W;
    I_s = (E_j == 1);  I_a = (E_j == 2);  
    I_hf = (W_j == 1);  I_lf = (W_j == 2);  I_vlf = (W_j == 3);  
    I_nw = (W_j == 4);

%-------------- Evaluate the gradient of log_pdf of alpha_c ---------------

    if (model.constraints(1) == "C*E") % c_j = c^(s,e) c^(s,c) c^(a,e) c^(a,c)
%         M = kron([I_s I_a],[1 1]);
%         M = [I_s I_s I_a I_a];
%         grad_alpha_j{1} = sum(LBA_pdf_j.grad_b.*M,1);
        grad_alpha_j{1} = [sum(LBA_pdf_j.grad_b.*I_s) sum(LBA_pdf_j.grad_b.*I_a)];           
    elseif (model.constraints(1) == "E") % c_j = c^(s) c^(a)   
        M = [I_s I_a];
        grad_alpha_j{1} = sum(sum(LBA_pdf_j.grad_b,2).*M);
    elseif (model.constraints(1) == "C") % c_j = c^(e) c^(c)
        grad_alpha_j{1} = sum(LBA_pdf_j.grad_b,1);   
    elseif (model.constraints(1) == "1")     
        grad_alpha_j{1} = sum(LBA_pdf_j.grad_b,'all');
    end
    grad_alpha_j{1} = grad_alpha_j{1} .*z_j{1};
    
%-------------- Evaluate the gradient of log_pdf of alpha_A ---------------
    grad_alpha_A = LBA_pdf_j.grad_b + LBA_pdf_j.grad_A;
    if (model.constraints(2) == "C*E") 
%         M = kron([I_s I_a],[1 1]);
%         M = [I_s I_s I_a I_a];
%         grad_alpha_j{1} = sum(grad_alpha_A.*M,1);
        grad_alpha_j{2} = [sum(grad_alpha_A.*I_s) sum(grad_alpha_A.*I_a)];         
    elseif (model.constraints(2) == "E")  
        M = [I_s I_a];
        grad_alpha_j{2} = sum(sum(grad_alpha_A,2).*M);
    elseif (model.constraints(2) == "C") 
        grad_alpha_j{2} = sum(grad_alpha_A,1); 
    elseif (model.constraints(2) == "1")     
        grad_alpha_j{2} = sum(grad_alpha_A,'all');
    end
    grad_alpha_j{2} = grad_alpha_j{2} .*z_j{2};
    
%-------------- Evaluate the gradient of log_pdf of alpha_v ---------------
    if (model.constraints(3) == "C*W*E")            
%         temp1 = [I_hf I_lf I_vlf I_nw].*I_s;
%         temp2 = [I_hf I_lf I_vlf I_nw].*I_a;
%         M = [temp1 temp2];
%         grad_alpha_j{3} = sum(LBA_pdf_j.grad_v.*M,1); 
        I_hf_s = I_hf.*I_s;   I_lf_s = I_lf.*I_s;   I_vlf_s = I_vlf.*I_s; I_nw_s = I_nw.*I_s;
        I_hf_a = I_hf.*I_a;   I_lf_a = I_lf.*I_a;   I_vlf_a = I_vlf.*I_a; I_nw_a = I_nw.*I_a;
        grad_alpha_j{3} = [sum(LBA_pdf_j.grad_v.*I_hf_s,1),sum(LBA_pdf_j.grad_v.*I_lf_s,1),...
            sum(LBA_pdf_j.grad_v.*I_vlf_s,1),sum(LBA_pdf_j.grad_v.*I_nw_s,1),...
            sum(LBA_pdf_j.grad_v.*I_hf_a,1),sum(LBA_pdf_j.grad_v.*I_lf_a,1),...
            sum(LBA_pdf_j.grad_v.*I_vlf_a,1),sum(LBA_pdf_j.grad_v.*I_nw_a,1)]; 
    elseif (model.constraints(3) == "C*W")   
        M = [I_hf I_hf I_lf I_lf I_vlf I_vlf I_nw I_nw];
        grad_alpha_j{3} = sum(repmat(LBA_pdf_j.grad_v,1,4).*M);  
    elseif (model.constraints(3) == "C*E")  
%         M = kron([I_s I_a],[1 1]);
%         M = [I_s I_s I_a I_a];
%         grad_alpha_j{3} = sum(LBA_pdf_j.grad_v.*M,1);
        grad_alpha_j{3} = [sum(LBA_pdf_j.grad_v.*I_s) sum(LBA_pdf_j.grad_v.*I_a)];  
    elseif (model.constraints(3) == "W*E")   
        temp1 = [I_hf I_lf I_vlf I_nw].*I_s;
        temp2 = [I_hf I_lf I_vlf I_nw].*I_a;
        M = [temp1 temp2];
        grad_alpha_j{3} = sum(sum(LBA_pdf_j.grad_v,2).*M);
     elseif (model.constraints(3) == "W")   
%         M = [I_hf I_lf I_vlf I_nw];
%         grad_alpha_j{3} = sum(LBA_pdf_j.grad_v.*M,1);
        grad_alpha_j{3} = [sum(LBA_pdf_j.grad_v.*I_hf,'all'),...
            sum(LBA_pdf_j.grad_v.*I_lf,'all'), sum(LBA_pdf_j.grad_v.*I_vlf,'all'),...
            sum(LBA_pdf_j.grad_v.*I_nw,'all')];
    elseif (model.constraints(3) == "E")   
        M = [I_s I_a];
        grad_alpha_j{3} = sum(sum(LBA_pdf_j.grad_v,2).*M);
    elseif (model.constraints(3) == "C") 
        grad_alpha_j{3} = sum(LBA_pdf_j.grad_v,1);   
    elseif (model.constraints(3) == "1")     
        grad_alpha_j{3} = sum(LBA_pdf_j.grad_v,'all');
    end 
    grad_alpha_j{3} = grad_alpha_j{3}.*z_j{3};
    
%-------------- Evaluate the gradient of log_pdf wrt alpha_s --------------

    grad_alpha_j{4} = [];
    
%-------------- Evaluate the gradient of log_pdf of alpha_tau -------------

    if (model.constraints(5) == "E") 
        M = [I_s I_a];
        grad_alpha_j{5} = sum(sum(LBA_pdf_j.grad_tau,2).*M,1);
    elseif (model.constraints(5) == "1") 
        grad_alpha_j{5} = sum(LBA_pdf_j.grad_tau,'all');
    end
    grad_alpha_j{5} = grad_alpha_j{5}.*z_j{5};    

end