function output  = LBA_pdf(RE,RT,b,A,v,s,tau,gradient)
%% DESCRIPTION: 
% This is the LBA density function 
% INPUT: Data: c (response choice) & t (response time) are column vectors of the same length n
%        Parameters: b,A,v,s & tau are n by 2 matrices (2 columns, one for each
%        accumulator vectors)
%        gradient = "true" => evaluate and return the gradient of log LBA
% OUTPUT: .log = log(LBA(c,t));
%         .grad_b = gradient of  log(LBA(c,t)) wrt b

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com


%Step 1: match the observed response with the accumulator
    ind_accumulator_2 = (RE == 2);   ind_accumulator_1 = (RE == 1);
    b_c = ind_accumulator_2.*b(:,2) + ind_accumulator_1.*b(:,1); 
    A_c = ind_accumulator_2.*A(:,2) + ind_accumulator_1.*A(:,1);
    v_c = ind_accumulator_2.*v(:,2) + ind_accumulator_1.*v(:,1); 
    s_c = ind_accumulator_2.*s(:,2) + ind_accumulator_1.*s(:,1); 
    tau_c = ind_accumulator_2.*tau(:,2) + ind_accumulator_1.*tau(:,1); 
    
    b_k = ind_accumulator_2.*b(:,1) + ind_accumulator_1.*b(:,2); 
    A_k = ind_accumulator_2.*A(:,1) + ind_accumulator_1.*A(:,2);
    v_k = ind_accumulator_2.*v(:,1) + ind_accumulator_1.*v(:,2); 
    s_k = ind_accumulator_2.*s(:,1) + ind_accumulator_1.*s(:,2); 
    tau_k = ind_accumulator_2.*tau(:,1) + ind_accumulator_1.*tau(:,2); 
    
%Step 2: Compute f_c (c,t) and adjust the gradient wrt v1 and v2
    %accordingly
    f_c = pdf_c(RT,b_c,A_c,v_c,s_c,tau_c,gradient);
    
%Step 3: Compute F_c (c,t) and adjust the gradient wrt v1 and v2
    %accordingly
    F_k = CDF_c(RT,b_k,A_k,v_k,s_k,tau_k,gradient);

%Step 4: Compute the log of LBA 
    log_element_wise = log(f_c.value) + log(F_k.substract);
    output.log_element_wise = log_element_wise;
    output.log = sum(log_element_wise);
  
%Step 5: Compute the gradient of log p(y_k|z_k) = p(y_k1|.) ...
    %p(y_kn|.)      
    if (gradient == "yes")    

         grad_b_acc1 = f_c.grad_b.*ind_accumulator_1 + F_k.grad_b.*ind_accumulator_2;
         grad_A_acc1 = f_c.grad_A.*ind_accumulator_1 + F_k.grad_A.*ind_accumulator_2;
         grad_v_acc1 = f_c.grad_v.*ind_accumulator_1 + F_k.grad_v.*ind_accumulator_2;
         grad_s_acc1 = f_c.grad_s.*ind_accumulator_1 + F_k.grad_s.*ind_accumulator_2;
         grad_tau_acc1 = f_c.grad_tau.*ind_accumulator_1 + F_k.grad_tau.*ind_accumulator_2;     

         grad_b_acc2 = f_c.grad_b.*ind_accumulator_2 + F_k.grad_b.*ind_accumulator_1;
         grad_A_acc2 = f_c.grad_A.*ind_accumulator_2 + F_k.grad_A.*ind_accumulator_1;
         grad_v_acc2 = f_c.grad_v.*ind_accumulator_2 + F_k.grad_v.*ind_accumulator_1;
         grad_s_acc2 = f_c.grad_s.*ind_accumulator_2 + F_k.grad_s.*ind_accumulator_1;
         grad_tau_acc2 = f_c.grad_tau.*ind_accumulator_2 + F_k.grad_tau.*ind_accumulator_1;

         output.grad_b = [ grad_b_acc1 grad_b_acc2];
         output.grad_A = [ grad_A_acc1 grad_A_acc2];
         output.grad_v = [ grad_v_acc1 grad_v_acc2];
         output.grad_s = [ grad_s_acc1 grad_s_acc2];
         output.grad_tau = [grad_tau_acc1 grad_tau_acc2];
    end     

end