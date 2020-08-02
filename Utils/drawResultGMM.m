function pointsToPlot = drawResultGMM(b, stDim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw the trajectory and uncertainty ellipses
%
% Input:
%   plotFn: function handle which sets line data
%   b: the beliefs to plot
%   stDim: robot state space dimension
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L = 1;

% itp = round(linspace(1,L,50)); % indexes to plot

x = b(1:stDim,:);

pointsToPlot = [x(3,:) NaN;x(4,:) NaN];

Ne = 50;% number of points in ellipse drawing
inc= 2*pi/Ne;
phi= 0:inc:2*pi;
sigmaScale = 3;

% get covariances
% for i = itp
    
    Sigma = zeros(stDim,stDim);
%     
    for d = 1:stDim
        Sigma(:,d) = b(d*stDim+1:(d+1)*stDim);
    end
    
    ptemp = make_ellipse_GMM(x(3:4),Sigma(3:4,3:4), sigmaScale, phi);
    
    if isempty(ptemp) == 0
        pointsToPlot = [pointsToPlot ptemp];
    end
% end

% plotFn(pointsToPlot);

end

function p= make_ellipse_GMM(x,P,s, phi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make a single 2-D ellipse of s-sigmas
% over phi angle intervals
%
% Input:
%   x: mean
%   P: covariance matrix
%   s: confidence bound (1-sigma, 2-sigma etc)
%   phi: angles from o to 2*pi
%
% Output:
%   p: the points on perimiter of ellipse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if s == 2 % 95% confidence
    chi2 = 5.991;
elseif s == 3 % 99% confidence
    chi2 = 9.210;
else
    error('Unknown confidence bound for drawing error ellipse');
end

magnify = 1.0; % scale up drawing

C = chol(P);

p = [];

if isempty(C) == 0
    a = C'*magnify*sqrt(chi2)*[cos(phi); sin(phi)];
    
    p=[a(1,:)+x(1) NaN;a(2,:)+x(2) NaN];
else
    a=1;
end

end
