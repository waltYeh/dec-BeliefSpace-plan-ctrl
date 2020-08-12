function yesno = isStateValid_multiagent(xi,ri, xj, rj)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check if robot is in collision with obstacles
% Input:
%   x: robot state
%   map: obstacle map
%   varargin: robot radius to override default
%
% Output:
%   yesno: 1 if robot is not in collision (valid state)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dx = xj(1,:) - xi(1);% map.obstacles has cols store with center pt of obst
dy = xj(2,:) - xi(2);

c2c = sqrt(dx.^2 + dy.^2); % center to center, robot to obst               
c2c_min = ri + rj;
if any(c2c <= c2c_min)
    yesno = 0;
    return;
end
yesno = 1;

end