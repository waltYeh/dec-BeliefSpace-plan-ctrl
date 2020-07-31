function J = multiAgentFiniteDifference2(fun, D,idx,x, h)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simple finite-difference derivatives
% assumes the function fun() is vectorized
% Inputs:
%   fun: function to differentiate
%   x: point at which to diff
%   h: step-size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if nargin < 4
    h = 2^-17;
% end

[n_agent, n_bu, K]  = size(x);%4,8,41

J=zeros(n_agent,n_bu,n_bu,K);
% we consider dyn decoupled situation first
% incoming_nbrs_idces = predecessors(D,idx);
for j=idx%[idx,incoming_nbrs_idces]
        H = zeros(size(D.Nodes,1),n_bu,1+n_bu);

        H(j,:,2:end)       = h*eye(n_bu);
    H       = permute(H, [1 2 4 3]);
    X       = pp(x(:,:,:), H);
    %this makes the X as: first of 3rd-dim are still x, the rest n of this dim
    %are x adding h for each element, pp is just elementwise plus
    X       = reshape(X, size(D.Nodes,1),n_bu, 1,K*(n_bu+1));
    %the fun has concatenated x/b and u as input, next x/b as output, so we know
    %dim of x/b is clear
    Y       = fun(X);%X 4x8x1x369, Y 4x8x369
    %numel: num of elements
    m       = numel(Y)/(K*(n_bu+1))/size(D.Nodes,1);
    Y       = reshape(Y, size(D.Nodes,1),m, K, n_bu+1);
    % Y 4x8x41x9
    J_j       = pp(Y(:,:,:,2:end), -Y(:,:,:,1)) / h;
    %this means (Y(x+h)-Y(x))/h
    J_j       = permute(J_j, [1 2 4 3]);
    % 4x8x8x41
    J = J_j;
end
end
