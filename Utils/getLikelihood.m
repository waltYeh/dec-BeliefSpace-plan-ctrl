function l = getLikelihood(r, A)
% r is x-mu
l = exp(-r'*inv(A)*r/2)/sqrt(det(2.*pi.*A));