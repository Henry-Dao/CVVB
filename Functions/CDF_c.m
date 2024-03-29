function output = CDF_c(t,b,A,v,s,tau,gradient)
%% DESCRIPTION: 
% This is the function F_c(t)
% INPUT: t,b,A,v,s,tau are column vectors
%        gradient = "true" => evaluate and return the gradient of log( 1- F_c(t))
% OUTPUT: .func = the value of F_c(t);
%         .substract = 1 - F_k(t)
%         .grad_b = gradient of log( 1- F_c(t)) wrt to b !!!!
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com


    t_diff = t - tau;
    % ----------- shortcut computations
    w1 = (b - t_diff.*v)./(s.*t_diff); w2 = A./(s.*t_diff);
    x = normcdf(w1-w2); x1 = normpdf(w1-w2); 
    z = normcdf(w1);  z1 = normpdf(w1); 
    u1 = (b-A-t_diff.*v)./A;     u2 = (b-t_diff.*v)./A;

%% to avoid numerical error
    id = x>0.999;
    x(id,1) = 0.999;
    id = x<=0.001;
    x(id,1) = 0.001;
    
    id = z>0.999;
    z(id,1) = 0.999;
    id = z<0.001;
    z(id,1) = 0.001;


%%
    term = -u1.*x + u2.*z - x1./w2 + z1./w2; %1 - F_k(t)
    ind_Fk = term<=10^-50;      % to avoid numerically zero
    term(ind_Fk) = 10^-50;      % to avoid numerically zero

    output.substract = term;
    output.value = 1 - term;  

    %-------------- Compute the partial derivatives
    if (gradient == "yes")
        x2 = phi_prime(w1-w2);  z2 = phi_prime(w1); 
        w3 = ((b-A)./s)./t_diff.^2; %the derivative of (w1 - w2) wrt tau 

        grad_b = (1./A).*x  - z./A + (u1.*x1- u2.*z1 + (1./w2).*(x2-z2))./(s.*t_diff);
        grad_A = ( -x + x1.*A.*u1./(s.*t_diff) )./A - (A.*u1.*x)./A.^2 + u2.*z./A + (x2.*w2-x1+z1)./(s.*t_diff.*w2.^2);
        grad_v = -t_diff.*x./A + (u1.*x1 - z1.*u2 )./s + t_diff.*z./A + (x2-z2)./(s.*w2);

        w12_s = (t_diff.*v - b + A)./(t_diff.*s.^2); % derivative of w1-w2 wrt to s
        w1_s = (t_diff.*v - b)./(t_diff.*s.^2); % derivative of w1 wrt to s
        w2_s = -A./(t_diff.*s.^2);
        grad_s = u1.*x1.*w12_s - u2.*z1.*w1_s + ( (x2.*w12_s - z2.*w1_s).*w2 - (x1-z1).*w2_s )./w2.^2; 
        grad_tau = (v./A).*x + u1.*x1.*w3 - (v./A).*z - u2.*z1.*(b./(s.*t_diff.^2)) +...
            (x2.*w2.*w3 - x1.*A./(s.*t_diff.^2) )./w2.^2 - ( z2.*w2.*b./(s.*t_diff.^2) - z1.*(A./(s.*t_diff.^2)) )./w2.^2;



        output.grad_b = -grad_b./term; % NOTE: term = 1- F_k(t)
        output.grad_A = -grad_A./term; % NOTE: grads hare are of log ( 1 - F_k(t)) !!!
        output.grad_v = -grad_v./term;
        output.grad_s = -grad_s./term;
        output.grad_tau = -grad_tau./term;

    end
end


