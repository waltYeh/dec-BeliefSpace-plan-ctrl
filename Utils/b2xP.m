function [x, P] = b2xP(b, stDim)
x = b(1:stDim);
size_x = size(x);
if size_x(2)>size_x(1)
    x=transpose(x);
end
P = zeros(stDim, stDim); % covariance matrix
% Extract columns of principal sqrt of covariance matrix
% right now we are not exploiting symmetry
for d = 1:stDim
    P(:,d) = b(d*stDim+1:(d+1)*stDim);
end
end

