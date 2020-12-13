function output = likelihood_Mnemonic_Hybrid(model,alpha,data,gradient)
% DESCRIPTION: Created at 10:14 pm 14/06/2020
%% INPUT STRUCTURE
% theta = (alpha_1, ... ,alpha_J,mu,vech_C*,log a)
% model is the structure with .name = ["C*E","1","W*E","W*E","1"];
%                             .index = [4, 1 , 8, 7, 1]; number of parameters
% OUTPUT: .log = log p(y|theta);
%         .grad  = (alpha_1',...alpha_n',mu_alpha', vech(C*)',log(a)') '
% gradient wrt theta, stack into a column vector.
loga = 0; % log of the total likelihood

D_alpha = sum(model.index); % n_alpha = D_alpha = dimension of random effect alpha
J = length(data.RT); % number of subjects/participants
grad_alpha = zeros(D_alpha,J); % store all the gradients wrt alpha_j
grad_mu = zeros(D_alpha,1);
grad_loga = zeros(D_alpha,1);
for j = 1:J 
%% Match each observation to the correct set of parameters (b,A,v,s,tau)
    n_j = length(data.RT{j}); % number of observations of subject j
    ind_acc = (data.E{j} == "accuracy");     ind_speed = (data.E{j} == "speed");   
    ind_S_new = (data.S{j} == "new");   ind_S_old = (data.S{j} == "old"); 
    ind_R_new = (data.S{j} == "new");   ind_R_old = (data.S{j} == "old");
    ind_match = (data.M{j} == "TRUE");  ind_mismatch = (data.M{j} == "FALSE");
    
    ind_Snew_match = ind_S_new.*ind_match;   ind_Snew_mismatch = ind_S_new.*ind_mismatch;
    ind_Sold_match = ind_S_old.*ind_match;   ind_Sold_mismatch = ind_S_old.*ind_mismatch;
    
    
    
    
    c_j = exp(alpha(1:model.index(1),j))'; % C
    A_j = exp(alpha(model.index(1)+1:sum(model.index(1:2)),j))'; % A
    v_j = exp(alpha(sum(model.index(1:2))+1:sum(model.index(1:3)),j))'; % v
    s_j = exp(alpha(sum(model.index(1:3))+1:sum(model.index(1:4)),j))'; % S
    tau_j = exp(alpha(sum(model.index(1:4))+1:sum(model.index),j))'; % T0
    
    % ----------- The threshold parameter c -----------------
    if (model.name(1) == "R") % c_j =  c^(n), c^(o) 
        c_j_stack_f = ind_R_new*c_j(1) +  ind_R_old*c_j(2);
        c_j_stack_F = ind_R_new*c_j(2) +  ind_R_old*c_j(1);

    else % c_j =  c^(n,s), c^(o,s), c^(n,a), c^(o,a)
        ind_Rnew_speed = ind_R_new.*ind_speed;   ind_Rold_speed = ind_R_old.*ind_speed;
        ind_Rnew_acc = ind_R_new.*ind_acc;   ind_Rold_acc = ind_R_old.*ind_acc;  
        
        c_j_stack_f = ind_Rnew_speed*c_j(1) + ind_Rold_speed*c_j(2) + ind_Rnew_acc*c_j(3) + ind_Rold_acc*c_j(4);
        c_j_stack_F = ind_Rnew_speed*c_j(2) + ind_Rold_speed*c_j(1) + ind_Rnew_acc*c_j(4) + ind_Rold_acc*c_j(3);       
    end
%     c_j_stack = [ c_j_stack_f c_j_stack_F ];
    
    % ----------- The start point parameter A -----------------    
    if (model.name(2) == "1")  % A = A
        A_j_stack = repmat(A_j,n_j,1); 
    else % A = A^(s) A^(a)
        A_j_stack = ind_speed*A_j(1) + ind_acc*A_j(2);
%         A_j_stack = repmat(temp,1,2);
    end
    
    % ----------- The drift rate mean v -----------------    
    
    if (model.name(3) == "S*M")  % v = v^(n,m), v^(n,mm), v^(o,m), v^(o,mm)             
        v_j_stack_f = ind_Snew_match*v_j(1) + ind_Snew_mismatch*v_j(2) + ind_Sold_match*v_j(3) + ind_Sold_mismatch*v_j(4); 
        v_j_stack_F = ind_Snew_match*v_j(2) + ind_Snew_mismatch*v_j(1) + ind_Sold_match*v_j(4) + ind_Sold_mismatch*v_j(3); 
    else % v = v^(s,n,m), v^(s,n,mm), v^(s,o,m), v^(s,o,mm), v^(a,n,m), v^(a,n,mm), v^(a,o,m), v^(a,o,mm)
        v_j_stack_f = ind_speed.*(ind_Snew_match*v_j(1) + ind_Snew_mismatch*v_j(2) + ind_Sold_match*v_j(3) + ind_Sold_mismatch*v_j(4)) + ...
                      ind_acc.*(ind_Snew_match*v_j(5) + ind_Snew_mismatch*v_j(6) + ind_Sold_match*v_j(7) + ind_Sold_mismatch*v_j(8));
                  
        v_j_stack_F = ind_speed.*(ind_Snew_match*v_j(2) + ind_Snew_mismatch*v_j(1) + ind_Sold_match*v_j(4) + ind_Sold_mismatch*v_j(3)) + ...
                      ind_acc.*(ind_Snew_match*v_j(6) + ind_Snew_mismatch*v_j(5) + ind_Sold_match*v_j(8) + ind_Sold_mismatch*v_j(7));       
    end
%     v_j_stack = [v_j_stack_f v_j_stack_F];
    
    % ----------- The drift rate std s -----------------    
    % s_j = s^(mm) since s^(m) = 1
    s_j_stack_f = ind_match + ind_mismatch*s_j;
    s_j_stack_F = ind_match*s_j + ind_mismatch;
%     s_j_stack = [s_j_stack_f s_j_stack_F];
    
    % ----------- The non-decision time parameter tau -----------------    
    if (model.name(5) == "1")  % tau = tau
        tau_j_stack = repmat(tau_j,n_j,1); 
    else % tau = tau^(s) tau^(a)
        tau_j_stack = ind_speed*tau_j(1) + ind_acc*tau_j(2);
%         tau_j_stack = repmat(temp,1,2);
    end
    
    b_j_stack_f = c_j_stack_f + A_j_stack; % Convert theta_j{1} from C to B.
    b_j_stack_F = c_j_stack_F + A_j_stack; % Convert theta_j{1} from C to B.
    
%% Compute the gradients

    %Step 2: Compute f_c (c,t) and adjust the gradient wrt v1 and v2
        %accordingly
        f_c = pdf_c(data.RT{j},b_j_stack_f,A_j_stack,v_j_stack_f,s_j_stack_f,tau_j_stack,gradient);

    %Step 3: Compute F_c (c,t) and adjust the gradient wrt v1 and v2
        %accordingly
        F_k = CDF_c(data.RT{j},b_j_stack_F,A_j_stack,v_j_stack_F,s_j_stack_F,tau_j_stack,gradient);

    %Step 4: Compute the log of LBA 
        loga = loga + sum(log(f_c.value) + log(F_k.substract));
  
    %Step 5: Compute the gradient of log p(y_k|z_k) = p(y_k1|.) ...
        %p(y_kn|.)      
    if (gradient == "true")        
        %% Rearrange the gradients into correct positions
% --------------------- The threshold parameter c -----------------
        if (model.name(1) == "R") % c_j =  c^(n), c^(o) 
                % -- gradient of log LBA_pdf wrt natural parameters --------
            grad_b_new = ind_R_new.*f_c.grad_b + ind_R_old.*F_k.grad_b;
            grad_b_old = ind_R_new.*F_k.grad_b + ind_R_old.*f_c.grad_b;
                % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_c_new = sum(grad_b_new)*c_j(1);
            grad_alpha_c_old = sum(grad_b_old)*c_j(2);
            
            
            grad_alpha_j_c = [grad_alpha_c_new grad_alpha_c_old];         

        else % c_j =  c^(n,s), c^(o,s), c^(n,a), c^(o,a)
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_b_new_speed = ind_Rnew_speed.*f_c.grad_b + ind_Rold_speed.*F_k.grad_b;
            grad_b_old_speed = ind_Rnew_speed.*F_k.grad_b + ind_Rold_speed.*f_c.grad_b;
            
            grad_b_new_acc = ind_Rnew_acc.*f_c.grad_b + ind_Rold_acc.*F_k.grad_b;
            grad_b_old_acc = ind_Rnew_acc.*F_k.grad_b + ind_Rold_acc.*f_c.grad_b;
                % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_c_new_speed = sum(grad_b_new_speed)*c_j(1); 
            grad_alpha_c_old_speed = sum(grad_b_old_speed)*c_j(2);
            
            grad_alpha_c_new_acc = sum(grad_b_new_acc)*c_j(3);
            grad_alpha_c_old_acc = sum(grad_b_old_acc)*c_j(4);
            
            grad_alpha_j_c = [grad_alpha_c_new_speed grad_alpha_c_old_speed grad_alpha_c_new_acc grad_alpha_c_old_acc]; 
        end

% --------------------- The threshold parameter A ------------------------

        if (model.name(2) == "1")  % A = A
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_A = f_c.grad_A + F_k.grad_A;
            grad_b = f_c.grad_b +  F_k.grad_b;
                % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_j_A = sum(grad_A + grad_b)*A_j; 
        else % A = A^(s) A^(a)
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_A = f_c.grad_A + F_k.grad_A;  
            grad_b = f_c.grad_b +  F_k.grad_b;
               % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_A_speed = sum(ind_speed.*(grad_A + grad_b))*A_j(1);
            grad_A_acc = sum(ind_acc.*(grad_A + grad_b))*A_j(2);
            
            grad_alpha_j_A = [grad_A_speed grad_A_acc];
        end
        
% ----------------------- The drift rate mean v --------------------------    
    
        if (model.name(3) == "S*M")  % v = v^(n,m), v^(n,mm), v^(o,m), v^(o,mm) 
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_v_new_match = ind_Snew_match.*f_c.grad_v + ind_Snew_mismatch.*F_k.grad_v;
            grad_v_new_mismatch = ind_Snew_match.*F_k.grad_v + ind_Snew_mismatch.*f_c.grad_v;
            
            grad_v_old_match = ind_Sold_match.*f_c.grad_v + ind_Sold_mismatch.*F_k.grad_v;
            grad_v_old_mismatch = ind_Sold_match.*F_k.grad_v + ind_Sold_mismatch.*f_c.grad_v;
            
               % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_v_new_match = sum(grad_v_new_match)*v_j(1);
            grad_alpha_v_new_mismatch = sum(grad_v_new_mismatch)*v_j(2);
            
            grad_alpha_v_old_match = sum(grad_v_old_match)*v_j(3);
            grad_alpha_v_old_mismatch = sum(grad_v_old_mismatch)*v_j(4);
            
            grad_alpha_j_v = [grad_alpha_v_new_match grad_alpha_v_new_mismatch ...
                                grad_alpha_v_old_match grad_alpha_v_old_mismatch];
        else % v = v^(s,n,m), v^(s,n,mm), v^(s,o,m), v^(s,o,mm), v^(a,n,m), v^(a,n,mm), v^(a,o,m), v^(a,o,mm)
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_v_new_match_speed = ind_speed.*(ind_Snew_match.*f_c.grad_v + ind_Snew_mismatch.*F_k.grad_v);
            grad_v_new_mismatch_speed = ind_speed.*(ind_Snew_match.*F_k.grad_v + ind_Snew_mismatch.*f_c.grad_v);         
            grad_v_old_match_speed = ind_speed.*(ind_Sold_match.*f_c.grad_v + ind_Sold_mismatch.*F_k.grad_v);
            grad_v_old_mismatch_speed = ind_speed.*(ind_Sold_match.*F_k.grad_v + ind_Sold_mismatch.*f_c.grad_v);
            
            grad_v_new_match_acc = ind_acc.*(ind_Snew_match.*f_c.grad_v + ind_Snew_mismatch.*F_k.grad_v);
            grad_v_new_mismatch_acc = ind_acc.*(ind_Snew_match.*F_k.grad_v + ind_Snew_mismatch.*f_c.grad_v);         
            grad_v_old_match_acc = ind_acc.*(ind_Sold_match.*f_c.grad_v + ind_Sold_mismatch.*F_k.grad_v);
            grad_v_old_mismatch_acc = ind_acc.*(ind_Sold_match.*F_k.grad_v + ind_Sold_mismatch.*f_c.grad_v);
            
               % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_j_v = sum( [grad_v_new_match_speed, grad_v_new_mismatch_speed,...
                grad_v_old_match_speed, grad_v_old_mismatch_speed, grad_v_new_match_acc,...
                grad_v_new_mismatch_acc, grad_v_old_match_acc, grad_v_old_mismatch_acc]).*v_j;
        end
        
% ----------------------- The drift rate standard deviation s -------------------------- 
            % -- gradient of log LBA_pdf wrt natural parameters -------
        % s_j = s^(mm) since s^(m) = 1
        grad_s = ind_match.*F_k.grad_s + ind_mismatch.*f_c.grad_s;
            % -------- grad_alpha : Multiplied by the Jacobian --------
        grad_alpha_j_s = sum(grad_s)*s_j;

% ---------------------------- The non-decision time tau -------------------------- 

        if (model.name(5) == "1")  % tau = tau
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_tau = f_c.grad_tau + F_k.grad_tau;
                % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_alpha_j_tau = sum(grad_tau)*tau_j; 
        else % tau = tau^(s) tau^(a)
                % -- gradient of log LBA_pdf wrt natural parameters -------
            grad_tau = f_c.grad_tau + F_k.grad_tau;  
               % -------- grad_alpha : Multiplied by the Jacobian --------
            grad_tau_speed = sum(ind_speed.*grad_tau)*tau_j(1);
            grad_tau_acc = sum(ind_acc.*grad_tau)*tau_j(2);
            
            grad_alpha_j_tau = [grad_tau_speed grad_tau_acc];
        end
        
    %% store the gradients  
        grad_alpha(:,j) = [grad_alpha_j_c'; grad_alpha_j_A'; grad_alpha_j_v'; grad_alpha_j_s'; grad_alpha_j_tau'];
         
    end     
    
end   
%% output of the function

%output.func = func;
    output.log = loga;
    if gradient == "true"
        output.grad = [grad_alpha(:); grad_mu; grad_loga];
    end
% gradient of log p(y|theta) wrt theta
end