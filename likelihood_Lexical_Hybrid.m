function output = likelihood_Lexical_Hybrid(model,alpha,data,gradient)
% DESCRIPTION: Created at 3:00 pm 2/08/2020
% TESTED AGAINST 'likelihood_Lexical_v1.m' under many model variants, using 'test_derivative.m'
% WHAT'S NEW: - change in computing gradients. Multiply be the Jacobian
% after summing up (line 92 - )
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
% alpha = reshape(theta(1:D_alpha*J),D_alpha,J); % this matrix store all the random effects
grad_alpha = zeros(D_alpha,J); % store all the gradients wrt alpha_j
grad_mu = zeros(D_alpha,1);
% grad_vech_C = zeros(D_alpha*(D_alpha+1)/2,1);
grad_loga = zeros(D_alpha,1);
for j = 1:J 
%% Match each observation to the correct set of parameters (b,A,v,s,tau)
    n_j = length(data.RT{j}); % number of observations of subject j
    ind_acc = (data.E{j} == 2);     ind_speed = (data.E{j} == 1);   
    ind_hf = (data.W{j} == 1);  ind_lf = (data.W{j} == 2);  ind_vlf = (data.W{j} == 3);  ind_nw = (data.W{j} == 4);
    ind_CWE = (model.name == "C*W*E");
    ind_CE = (model.name == "C*E");  ind_CW = (model.name == "C*W");  ind_WE = (model.name == "W*E");
    ind_C = (model.name == "C");  ind_E = (model.name == "E");  ind_W = (model.name == "W");
    ind_1 = (model.name == "1");    ind_0 = (model.name == "0");
    
    alpha_j{1} = exp(alpha(1:model.index(1),j))'; % C
    alpha_j{2} = exp(alpha(model.index(1)+1:sum(model.index(1:2)),j))'; % A
    alpha_j{3} = exp(alpha(sum(model.index(1:2))+1:sum(model.index(1:3)),j))'; % v
    alpha_j{4} = [exp(alpha(sum(model.index(1:3))+1:sum(model.index(1:4)),j))' 1]; % S
    alpha_j{5} = exp(alpha(sum(model.index(1:4))+1:sum(model.index),j))'; % T0


    
 
    theta_j = cell(1,5);
    if (sum(ind_CWE)~=0) 
%         x = [ (c,w,speed) then (c,w,acc)]
        theta_j(ind_CWE) = cellfun(@(x) (x(1:2).*ind_hf + x(3:4).*ind_lf + x(5:6).*ind_vlf + x(7:8).*ind_nw).*ind_acc + ...
            (x(9:10).*ind_hf + x(11:12).*ind_lf + x(13:14).*ind_vlf + x(15:16).*ind_nw).*ind_speed,alpha_j(ind_CWE),'UniformOutput',false); 
    end
    
    if (any(ind_CE)~=0) 
%         x = [ (c,speed) (nw,acc)] 
        theta_j(ind_CE) = cellfun(@(x) x(1:2).*ind_acc + x(3:4).*ind_speed,alpha_j(ind_CE),'UniformOutput',false); 
    end
    
    if (any(ind_CW)~=0) 
%         x = [ (c,hf) ... (e,nw)]
        theta_j(ind_CW) = cellfun(@(x) x(1:2).*ind_hf + x(3:4).*ind_lf + x(5:6).*ind_vlf + x(7:8).*ind_nw,alpha_j(ind_CW),'UniformOutput',false); 
    end
    
    if (any(ind_WE)~=0) 
%         x = [(hf,speed) ... (nw, acc)]
        temp1 = [ind_hf ind_lf ind_vlf ind_nw].*ind_acc;
        temp2 = [ind_hf ind_lf ind_vlf ind_nw].*ind_speed;
        M = [temp1 temp2];
        theta_j(ind_WE) = cellfun(@(x) repmat(M*x',1,2),alpha_j(ind_WE),'UniformOutput',false); 
    end
    
    if (any(ind_W)~=0) 
        M = [ind_hf ind_lf ind_vlf ind_nw];
        theta_j(ind_W) = cellfun(@(x) repmat(M*x',1,2),alpha_j(ind_W),'UniformOutput',false); 
    end
    
    if (any(ind_C)~=0) 
        theta_j(ind_C) = cellfun(@(x) repmat(x,n_j,1),alpha_j(ind_C),'UniformOutput',false); 
    end
    
    if (any(ind_E)~=0) 
        M = [ind_acc ind_speed ];
        theta_j(ind_E) = cellfun(@(x) repmat(M*x',1,2),alpha_j(ind_E),'UniformOutput',false); 
    end
    
    if (any(ind_1)~=0) 
        theta_j(ind_1) = cellfun(@(x) repmat(x,n_j,2),alpha_j(ind_1),'UniformOutput',false); 
    end
    
    if (any(ind_0)~=0) 
        theta_j{ind_0} = ones(n_j,2); 
    end
    
    theta_j{1} = theta_j{1} + theta_j{2}; % Convert theta_j{1} from C to B.
    
%% Compute the gradients

        LBA_j = LBA_pdf(data.RE{j},data.RT{j},theta_j{1},theta_j{2},theta_j{3},theta_j{4},theta_j{5},gradient);
        loga = loga + LBA_j.log;
    if gradient == "true"
        grad_LBA_j = cell(1,5);
        grad_LBA_j{1} = LBA_j.grad_b; % grad_b
        grad_LBA_j{2} = LBA_j.grad_b + LBA_j.grad_A; % grad_A + grad_b
        grad_LBA_j{3} = LBA_j.grad_v; % grad_v 
        grad_LBA_j{4} = LBA_j.grad_s; % grad_s
        grad_LBA_j{5} = LBA_j.grad_tau; % grad_tau

    %% Rearrange the gradients into correct positions
        grad_alpha_j = cell(1,5);
        if (sum(ind_CWE)~=0) 
    %         x = [ (c,w,speed) then (c,w,acc)]
            grad_alpha_j(ind_CWE) = cellfun(@(x,y) [sum(x.*ind_hf.*ind_acc) sum(x.*ind_lf.*ind_acc) sum(x.*ind_vlf.*ind_acc) ...
                sum(x.*ind_nw.*ind_acc) sum(x.*ind_hf.*ind_speed) sum(x.*ind_lf.*ind_speed) sum(x.*ind_vlf.*ind_speed) ...
                sum(x.*ind_nw.*ind_speed) ].*y,grad_LBA_j(ind_CWE),alpha_j(ind_CWE),'UniformOutput',false); 
        end

        if (any(ind_CE)~=0) 
    %         x = [ (c,speed) (nw,acc)] 
            grad_alpha_j(ind_CE) = cellfun(@(x,y) [sum(x.*ind_acc) sum(x.*ind_speed)].*y,grad_LBA_j(ind_CE),alpha_j(ind_CE),'UniformOutput',false); 
        end

        if (any(ind_CW)~=0) 
            M = [ind_hf ind_hf ind_lf ind_lf ind_vlf ind_vlf ind_nw ind_nw];
            grad_alpha_j(ind_CW) = cellfun(@(x,y) sum(repmat(x,1,4).*M).*y,grad_LBA_j(ind_CW),alpha_j(ind_CW),'UniformOutput',false); 
    %         grad_alpha_j(ind_CW) = cellfun(@(x,y) [sum(x.*ind_hf) sum(x.*ind_lf) sum(x.*ind_vlf) sum(x.*ind_nw)].*y,grad_LBA_j(ind_CW),alpha_j(ind_CW),'UniformOutput',false); 
        end

        if (any(ind_WE)~=0) 
            temp1 = [ind_hf ind_lf ind_vlf ind_nw].*ind_acc;
            temp2 = [ind_hf ind_lf ind_vlf ind_nw].*ind_speed;
            M = [temp1 temp2];
            grad_alpha_j(ind_WE) = cellfun(@(x,y) sum(sum(x,2).*M).*y,grad_LBA_j(ind_WE),alpha_j(ind_WE),'UniformOutput',false);
    %         grad_alpha_j(ind_WE) = cellfun(@(x,y) [sum(x.*ind_hf.*ind_speed,'all') sum(x.*ind_lf.*ind_speed,'all') ...
    %             sum(x.*ind_vlf.*ind_speed,'all') sum(x.*ind_nw.*ind_speed,'all') sum(x.*ind_hf.*ind_acc,'all') sum(x.*ind_lf.*ind_acc,'all') ...
    %             sum(x.*ind_vlf.*ind_acc,'all') sum(x.*ind_nw.*ind_acc,'all')].*y,grad_LBA_j(ind_WE),alpha_j(ind_WE),'UniformOutput',false); 
        end

        if (any(ind_W)~=0) 
            M = [ind_hf ind_lf ind_vlf ind_nw];
            grad_alpha_j(ind_W) = cellfun(@(x,y) [sum(x.*ind_hf,'all') sum(x.*ind_lf,'all') sum(x.*ind_vlf,'all') sum(x.*ind_nw,'all')].*y,grad_LBA_j(ind_W),alpha_j(ind_W),'UniformOutput',false); 
        end

        if (any(ind_C)~=0) 
            grad_alpha_j(ind_C) = cellfun(@(x,y) sum(x).*y,grad_LBA_j(ind_C),alpha_j(ind_C),'UniformOutput',false); 
        end

        if (any(ind_E)~=0) 
            M = [ind_acc ind_speed];
            grad_alpha_j(ind_E) = cellfun(@(x,y) sum(sum(x,2).*M).*y,grad_LBA_j(ind_E),alpha_j(ind_E),'UniformOutput',false); 
    %         grad_alpha_j(ind_E) = cellfun(@(x,y) [sum(x.*ind_acc,'all') sum(x.*ind_speed,'all')].*y,grad_LBA_j(ind_E),alpha_j(ind_E),'UniformOutput',false); 
        end

        if (any(ind_1)~=0) 
            grad_alpha_j(ind_1) = cellfun(@(x,y) sum(x,'all')*y,grad_LBA_j(ind_1),alpha_j(ind_1),'UniformOutput',false); 
        end

    %     if (any(ind_0)~=0) 
    %         grad_alpha_j(ind_0) = []; 
    %     end

    %% Remove the last element of grad_alpha_j{4} (means s = 1 ) for identifiability
        if (ind_0(4)==0) grad_alpha_j{4}(end) = [];
        end
    %% store the gradients  
        grad_alpha(:,j) = [grad_alpha_j{1}'; grad_alpha_j{2}'; grad_alpha_j{3}'; grad_alpha_j{4}'; grad_alpha_j{5}'];
    end   
%% output of the function
end
%output.func = func;
    output.log = loga;
    if gradient == "true"
        output.grad = [grad_alpha(:); grad_mu; grad_loga];
    end
% gradient of log p(y|theta) wrt theta
end