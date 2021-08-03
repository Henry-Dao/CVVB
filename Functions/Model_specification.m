if exp_name == "Mnemonic"
    M = 16; % total of competing models
    model = cell(1,M);
    constraints_c = ["R","E*R"];
    constraints_A = ["1","E"];
    constraints_v = ["S*M", "E*S*M"];
    constraints_s = ["M"];
    constraints_tau = constraints_A;
    num_ind = [1 2 4 8]; % number of random effects per subject
    m_ind = 1;
    for c_ind = 1:2
        for A_ind = 1:2
            for v_ind = 1:2
                for tau_ind = 1:2
                    model{m_ind}.constraints = [constraints_c(c_ind), constraints_A(A_ind), constraints_v(v_ind), constraints_s, constraints_tau(tau_ind)];
                    model{m_ind}.dim = [num_ind(c_ind+1), num_ind(A_ind), num_ind(v_ind+2), 1, num_ind(tau_ind)];
                    
                    D_alpha = sum(model{m_ind}.dim);
                    model{m_ind}.prior_par.mu= zeros(D_alpha,1); 
                    model{m_ind}.prior_par.cov = eye(D_alpha); 
                    model{m_ind}.prior_par.v_a = 2; 
                    model{m_ind}.prior_par.A_d = ones(D_alpha,1); 
                    m_ind = m_ind + 1;
                end
            end
        end
    end
elseif exp_name == "Lexical"
    M = 256; % total of competing models
    model = cell(1,M);
    constraints_b = ["1","C","E","C*E"];
    constraints_A = constraints_b;
    constraints_v = ["1","C","E","W","C*E","C*W","W*E","C*W*E"];
    constraints_tau = ["1","E"];
    num_ind = [1 2 2 4 4 8 8 16]; % number of random effects per subject
    m_ind = 1;
    for v_ind = 1:8
        for A_ind = 1:4
            for b_ind = 1:4
                for tau_ind = 1:2
                    model{m_ind}.constraints = [constraints_b(b_ind), constraints_A(A_ind), constraints_v(v_ind), "0", constraints_tau(tau_ind)];
                    model{m_ind}.dim = [num_ind(b_ind), num_ind(A_ind), num_ind(v_ind), 0, num_ind(tau_ind)];
                    
                    D_alpha = sum(model{m_ind}.dim);
                    model{m_ind}.prior_par.mu= zeros(D_alpha,1); 
                    model{m_ind}.prior_par.cov = eye(D_alpha); 
                    model{m_ind}.prior_par.v_a = 2; 
                    model{m_ind}.prior_par.A_d = ones(D_alpha,1); 
                    m_ind = m_ind + 1;
                end
            end
        end
    end
elseif exp_name == "Forstmann"
    M = 27; % total of competing models
    model = cell(1,M);
    model_ind = [1 1 1; 1 1 2; 1 1 3; 1 2 3; 1 2 2; 1 2 1; 1 3 1; 1 3 2; 
        1 3 3;  2 3 3; 2 3 2; 2 3 1; 2 2 1; 2 2 2; 2 2 3; 2 1 3; 2 1 2; 2 1 1;
        3 1 1; 3 1 2; 3 1 3; 3 2 3; 3 2 2; 3 2 1; 3 3 1; 3 3 2; 3 3 3];% each row of matrix z is [z_c, z_v, z_tau
    
    constraints = ["1", "2", "3"];
    for i = 1:M
        ind = model_ind(i,:);
        model{i}.constraints = [constraints(ind(1)), "1", constraints(ind(2)), "0", constraints(ind(3))];
        model{i}.dim = zeros(5,1);
        model{i}.dim(2) = 1;
        model{i}.dim([1 3 5]) = ind;
        model{i}.dim(3) = 2*model{i}.dim(3);
        
        D_alpha = sum(model{i}.dim);
        model{i}.prior_par.mu= zeros(D_alpha,1); 
        model{i}.prior_par.cov = eye(D_alpha); 
        model{i}.prior_par.v_a = 2; 
        model{i}.prior_par.A_d = ones(D_alpha,1); 
    end
end