function y = q_VAFC(theta,mu,B,d)
% NOTE: This version is the same as q_gaussian_factor_covariance.m, except
% that the input is (mu,B,d), not lambda as in the latter
% INPUT: lambda = (mu,B,d), theta ~ N(mu,covmat) 
%                 with covmat = lambda.B*lambda.B' + diag(lambda.d.^2);
% OUTPUT: y.log = log (q_lambda)
%         y.theta = inv(BB'+D^2)*(Bz + d.*eps) %gradient of q wrt beta

[m p] = size(B);
covmat = B*B' + diag(d.^2);     term = theta - mu;
D_inv2 = diag(d.^(-2));
precision_matrix = D_inv2 - D_inv2*B/(eye(p) + B'*D_inv2*B)*B'*D_inv2;
y.log = -0.5*p*log(2*pi) - 0.5*logdet(covmat) - 0.5*term'/covmat*term;
y.grad = - precision_matrix*term;
end
