function output = vech_inv(b,d)
% INPUT: b is a vector ; d = is the dimension of the output square matrix
% OUTPUT: A lower triangular matrix B such as vech(B) = b 
A = ones(d); %create a dxd matrix of 1s
lower_A = tril(A); %get lower triangular part of A
ind = find(lower_A); %find nonzero elements of lower_A, basically this get linear index of elements below and on diagonal of A !!!
B = zeros(d);
B(ind)=b;
output = B;
end