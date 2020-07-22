function c = costAssistingRobot(b, u, horizon, stDim,components_amount)
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

c = zeros(1,size(b,2));

for i=1:size(b,2)
%     if isempty(varargin)
        c(i) =  evaluateCost(b(:,i),u(:,i), stDim,components_amount, horizon);
%     else
%         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, horizon, stateValidityChecker, varargin{1});
%     end
end

end

function cost = evaluateCost(b, u, stDim, components_amount, L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cost for a states according to cost model given in Section 6 
% of Van Den Berg et al. IJRR 2012
%
% Input:
%   b: Current belief vector
%   u: Control
%   goal: target state
%   stDim: State dimension
%   L: Number of steps in horizon
%   stateValidityChecker: checks if state is in collision or not
% Outputs:
%   c: cost estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

final = isnan(u(1,:));
u(:,final)  = 0;

ctrlDim = size(u,1);

[x, P, w] = b2xPw(b, stDim, 2);

Q_t = 10*eye(stDim); % penalize uncertainty
R_t = diag([0.2, 0.01, 0.2, 0.2,0.001,0.001]); % penalize control effort
Q_t_err = 1.0*eye(2);
Q_l = 10*L*eye(2); % penalize terminal error
Qcov_l = 10*L*eye(4); % penalize terminal covar
w_cc = 1.0; % penalize collision


component_cost = zeros(components_amount,1);
cost = 0;
% deviation from goal
for i_comp=1:components_amount
    delta_x = x{i_comp}(1:2)-x{i_comp}(3:4);
    % collision cost
    cc = 0;

    % State Cost
    sc = 0;

    % information cost
    ic = 0;

    % control cost
    uc = 0;

    % final cost
    if any(final)

      sc = delta_x'*Q_l*delta_x;

      ic = trace(P{i_comp}*Qcov_l*P{i_comp});

    else
      sc = delta_x'*Q_t_err*delta_x;
      ic = trace(P{i_comp}*Q_t*P{i_comp});

      uc = u'*R_t*u;


    end
    component_cost(i_comp) = sc + ic + uc + w_cc*cc;
    cost = cost + component_cost(i_comp) * w(i_comp)^2;
end
end