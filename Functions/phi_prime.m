function output = phi_prime(x)
% phi_prime is the first derivative of phi (standard normal density )
output = -normpdf(x).*x;
end