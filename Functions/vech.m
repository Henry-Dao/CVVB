function output = vech(A)
% This function gives vech(A) where A is a square matrix
[d,~] = size(A);
ind = tril(ones(d));
output = A(ind==1);

end