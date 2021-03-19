function output = vech(A)
[d,~] = size(A);
% b = A(:,1);
% for i=2:d
%     b = [b; A(i:end,i)];
% end
% output = b;
% below is a better version

ind = tril(ones(d));
output = A(ind==1);

end