function output = likelihood_Forstmann_Hybrid(model,alpha,data,gradient)
% DESCRIPTION: Created at 13:14 pm 13/07/2020.
% TESTING COMPARED WITH likelihood_Forstmann_old (see test_derivative) !
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
grad_loga = zeros(D_alpha,1);
for j = 1:J 
%% Match each observation to the correct set of parameters (b,A,v,s,tau)
    n_j = length(data.RT{j}); % number of observations of subject j
    ind_acc = (data.E{j} == 1);   ind_neutral = (data.E{j} == 2);     ind_speed = (data.E{j} == 3);    
    
    
    alpha_j{1} = exp(alpha(1:model.index(1),j))'; % C
    alpha_j{2} = exp(alpha(model.index(1)+1:sum(model.index(1:2)),j))'; % A
    alpha_j{3} = exp(alpha(sum(model.index(1:2))+1:sum(model.index(1:3)),j))'; % v
    alpha_j{4} = [exp(alpha(sum(model.index(1:3))+1:sum(model.index(1:4)),j))' 1]; % S
    alpha_j{5} = exp(alpha(sum(model.index(1:4))+1:sum(model.index),j))'; % T0
    
    theta_j = cell(1,5);
            % Duplicate parameter c accordingly to model
    if (model.name(1)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        theta_j(1) = cellfun(@(x) repmat(M*x',1,2),alpha_j(1),'UniformOutput',false); 
    elseif (model.name(1)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        theta_j(1) = cellfun(@(x) repmat(M*x',1,2),alpha_j(1),'UniformOutput',false); 
    else
        theta_j(1) = cellfun(@(x) repmat(x,n_j,2),alpha_j(1),'UniformOutput',false);
    end
    
            % Duplicate parameter A accordingly to model
    theta_j(2) = cellfun(@(x) repmat(x,n_j,2),alpha_j(2),'UniformOutput',false);
    
            % Duplicate parameter v accordingly to model
    if (model.name(3) == "3")
        theta_j(3) = cellfun(@(x) ind_acc.*x(1:2) + ind_neutral.*x(3:4) + ind_speed.*x(5:6),alpha_j(3),'UniformOutput',false);
    elseif (model.name(3) == "2")
        theta_j(3) = cellfun(@(x) (ind_acc + ind_neutral).*x(1:2) + ind_speed.*x(3:4),alpha_j(3),'UniformOutput',false);
    else
        theta_j(3) = cellfun(@(x) repmat(x,n_j,1),alpha_j(3),'UniformOutput',false);
    end

            % Duplicate parameter sv accordingly to model
    theta_j{4} = ones(n_j,2);
    
            % Duplicate parameter tau accordingly to model
    if (model.name(5)== "3") 
        M = [ind_acc ind_neutral ind_speed ];
        theta_j(5) = cellfun(@(x) repmat(M*x',1,2),alpha_j(5),'UniformOutput',false); 
    elseif (model.name(5)== "2") 
        M = [(ind_acc + ind_neutral) ind_speed ];
        theta_j(5) = cellfun(@(x) repmat(M*x',1,2),alpha_j(5),'UniformOutput',false); 
    else
        theta_j(5) = cellfun(@(x) repmat(x,n_j,2),alpha_j(5),'UniformOutput',false);
    end
    
    
    theta_j{1} = theta_j{1} + theta_j{2}; % Convert theta_j{1} from C to B.
    
%% Compute the gradients
    LBA_j = LBA_pdf(data.RE{j},data.RT{j},theta_j{1},theta_j{2},theta_j{3},theta_j{4},theta_j{5},gradient);
    loga = loga + LBA_j.log;
    if gradient == "true"
% Multiply by the Jacobian to get gradient wrt to alpha   
        grad_LBA_j = cell(1,5);
        grad_LBA_j{1} = LBA_j.grad_b.*(theta_j{1}-theta_j{2}); % grad_alpha_c
        grad_LBA_j{2} = (LBA_j.grad_b + LBA_j.grad_A).*theta_j{2}; % grad_alpha_A
        grad_LBA_j{3} = LBA_j.grad_v.*theta_j{3}; % grad_alpha_v 
%         grad_LBA_j{4} = LBA_j.grad_s.*theta_j{4}; % grad_alpha_s
        grad_LBA_j{5} = LBA_j.grad_tau.*theta_j{5}; % grad_alpha_tau
    
%% Rearrange the gradients into correct positions
        grad_alpha_j = cell(1,5);
        
                % Rearranage the gradient of log_pdf of alpha_c
                
        if (model.name(1)== "3") 
            grad_alpha_j(1) = cellfun(@(x) [sum(x.*ind_acc,'all') sum(x.*ind_neutral,'all') sum(x.*ind_speed,'all')],grad_LBA_j(1),'UniformOutput',false); 
        elseif (model.name(1)== "2") 
            grad_alpha_j(1) = cellfun(@(x) [sum(x.*(ind_acc + ind_neutral),'all') sum(x.*ind_speed,'all')],grad_LBA_j(1),'UniformOutput',false);
        else
            grad_alpha_j(1) = cellfun(@(x) sum(x,'all'),grad_LBA_j(1),'UniformOutput',false);
        end

                % Rearranage the gradient of log_pdf of alpha_A
                
        grad_alpha_j{2} = sum(grad_LBA_j{2},'all');
        
        
                 % Rearranage the gradient of log_pdf of alpha_v
                
        if (model.name(3)== "3") 
            grad_alpha_j(3) = cellfun(@(x) [sum(x.*ind_acc) sum(x.*ind_neutral) sum(x.*ind_speed)],grad_LBA_j(3),'UniformOutput',false); 
        elseif (model.name(3)== "2") 
            grad_alpha_j(3) = cellfun(@(x) [sum(x.*(ind_acc + ind_neutral)) sum(x.*ind_speed)],grad_LBA_j(3),'UniformOutput',false);
        else
            grad_alpha_j(3) = cellfun(@(x) sum(x),grad_LBA_j(3),'UniformOutput',false);
        end 

                % Rearranage the gradient of log_pdf of alpha_A
                
        grad_alpha_j{4} = []; 
        
        
                % Rearranage the gradient of log_pdf of alpha_tau
                
        if (model.name(5)== "3") 
            grad_alpha_j(5) = cellfun(@(x) [sum(x.*ind_acc,'all') sum(x.*ind_neutral,'all') sum(x.*ind_speed,'all')],grad_LBA_j(5),'UniformOutput',false); 
        elseif (model.name(5)== "2") 
            grad_alpha_j(5) = cellfun(@(x) [sum(x.*(ind_acc + ind_neutral),'all') sum(x.*ind_speed,'all')],grad_LBA_j(5),'UniformOutput',false);
        else
            grad_alpha_j(5) = cellfun(@(x) sum(x,'all'),grad_LBA_j(5),'UniformOutput',false);
        end
       

%% Remove the last element of grad_alpha_j{4} (means s = 1 ) for identifiability
%         if (ind_0(4)==0) grad_alpha_j{4}(end) = [];
%         end
%% store the gradients  
        grad_alpha(:,j) = [grad_alpha_j{1}'; grad_alpha_j{2}'; grad_alpha_j{3}'; grad_alpha_j{4}'; grad_alpha_j{5}'];
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