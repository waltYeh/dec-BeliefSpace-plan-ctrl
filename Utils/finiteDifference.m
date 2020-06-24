function J = finiteDifference(fun, x, h)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simple finite-difference derivatives
% assumes the function fun() is vectorized
% Inputs:
%   fun: function to differentiate
%   x: point at which to diff
%   h: step-size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    h = 2^-17;
end

[n, K]  = size(x);
H       = [zeros(n,1) h*eye(n)];
H       = permute(H, [1 3 2]);
X       = pp(x, H);
%this makes the X as: first of 3rd-dim are still x, the rest n of this dim
%are x adding h for each element, pp is just elementwise plus
X       = reshape(X, n, K*(n+1));
%the fun has concatenated x/b and u as input, next x/b as output, so we know
%dim of x/b is clear
Y       = fun(X);
%numel: num of elements
m       = numel(Y)/(K*(n+1));
Y       = reshape(Y, m, K, n+1);
J       = pp(Y(:,:,2:end), -Y(:,:,1)) / h;
%this means (Y(x+h)-Y(x))/h
J       = permute(J, [1 3 2]);
end