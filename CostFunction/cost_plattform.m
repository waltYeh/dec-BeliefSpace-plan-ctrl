function c = cost_plattform(b, u,...
    horizon,components_amount)
% one step cost, not the whole cost horizon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cost for vector of states according to cost model given in Section 6 
% of Van Den Berg et al. IJRR 2012
%
% Input:
%   b: Current belief vector
%   u: Control
%   goal: target state
%   L: Total segments
%   stDim: state space dimension for robot
%   stateValidityChecker: checks if state is in collision or not
% Outputs:
%   c: cost estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = zeros(1,size(b,3));
% if size(b,3)>1  
% % % size is n_b * 1 * ((n_b+n_u+1)horizon) for deri before backward path
% % % size of u is n_u * 1 * ((n_b+n_u+1)horizon) with NaN at the end of
% % % horizons
%     keyboard
% end
% size is n_agent * n_b * parallel_alpha for forward path
% incoming_nbrs_idces = predecessors(D,idx);
for j=1:size(b,2)
%     if isempty(varargin)
%     b_this_for_paral = cell(size(b));
%     u_this_for_paral = cell(size(u));
%     for i = incoming_nbrs_idces
%         b_this_for_paral{i} = b{i}(:,j);
%         u_this_for_paral{i} = u{i}(:,j);
%     end
%     b_this_for_paral{idx} = b{idx}(:,j);
%     u_this_for_paral{idx} = u{idx}(:,j);

    b_parallel = b(:,j);
    u_parallel = u(:,j);
    c(j) =  evaluateCost(b_parallel,u_parallel,...
         horizon,components_amount);
%     else
%         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, L, stateValidityChecker, varargin{1});
%     end
end

end

function cost = evaluateCost(b, u,  ...
    L, components_amount)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cost for a states according to cost model given in Section 6 
% of Van Den Berg et al. IJRR 2012
%
% Input:
%   b: Current belief vector 4x6
%   u: Control 4x2
%   goal: target state
%   stDim: State dimension
%   L: Number of steps in horizon
%   stateValidityChecker: checks if state is in collision or not
% Outputs:
%   c: cost estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cost = 0;
final = isnan(u(1,:));
% u(:,final)  = 0;
stDim = 4;
[x_idx, P_idx, w] = b2xPw(b(:,1), stDim, components_amount);

% u{idx}(:,final)  = 0;
R_t = diag([0.2, 4.0, 0.2, 0.2,0.1,0.1]);
Qerr_l = 10*L*eye(2);
Qerr_t = 0.05*eye(2);
Qcov_l = 100000000*eye(4); % penalize terminal covar
Qcov_l(1,1) = 0;
Qcov_l(2,2) = 0;
component_cost = zeros(components_amount,1);
cost = 0;
for i_comp=1:components_amount
    delta_x = x_idx{i_comp}(1:2)-x_idx{i_comp}(3:4);
    % collision Cost
    cc = 0;
    w_cc = 1.0;
    % State Cost
    sc = 0;
    % information cost
    ic = 0;
    % control cost
    uc = 0;
    if any(final)
        sc = delta_x'*Qerr_l*delta_x;
        ic = trace(P_idx{i_comp}*Qcov_l*P_idx{i_comp});
    else
        uc = u'*R_t*u;
        sc = delta_x'*Qerr_t*delta_x;
    end
    component_cost(i_comp) = sc + ic + uc + w_cc*cc;
    
    cost = cost + component_cost(i_comp) * w(i_comp)^2;
end
end
