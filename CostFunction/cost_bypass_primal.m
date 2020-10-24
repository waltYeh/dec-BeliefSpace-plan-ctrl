function c = cost_bypass_primal(D, idx, b, u,last_com_t,lam,rho,horizon,  ...
    stateValidityChecker)
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
% size is n_agent * n_b for forward path
% incoming_nbrs_idces = predecessors(D,idx);
for j=1:size(b{idx},2)
%     if isempty(varargin)
%     b_this_for_paral = cell(size(b));
%     u_this_for_paral = cell(size(u));
%     for i = incoming_nbrs_idces
%         b_this_for_paral{i} = b{i}(:,j);
%         u_this_for_paral{i} = u{i}(:,j);
%     end
%     b_this_for_paral{idx} = b{idx}(:,j);
%     u_this_for_paral{idx} = u{idx}(:,j);
    b_parallel = b;
    u_parallel = u;
    for i = 1:size(D.Nodes,1)
        b_parallel{i} = b{i}(:,j);
        u_parallel{i} = u{i}(:,j);
    end
    c(j) =  evaluateCost(D, idx, b_parallel,u_parallel,last_com_t,...
        lam.lam_d(:,:,j),lam.lam_up(:,:,j),lam.lam_c(:,:,j),lam.lam_w(:,:,j),...
        rho.rho_d,rho.rho_up,rho.rho_w,horizon, ...
        stateValidityChecker);
%     else
%         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, L, stateValidityChecker, varargin{1});
%     end
end

end

function cost = evaluateCost(D, idx, b, u, last_com_t,lam_di,lam_up,lam_c,lam_w,rho_d,rho_up,rho_w, ...
    L, stateValidityChecker)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cost for a states according to cost model given in Section 6 
% of Van Den Berg et al. IJRR 2012
%
% Input:
%   b: Current belief vector 4x2
%   u: Control 4x6
%   goal: target state
%   stDim: State dimension
%   L: Number of steps in horizon
%   stateValidityChecker: checks if state is in collision or not
% Outputs:
%   c: cost estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cost = 0;
incoming_nbrs_idces = predecessors(D,idx)';
[eid,nid] = inedges(D,idx);
final = isnan(u{idx}(1,:));
% u(:,final)  = 0;
for j = [idx, incoming_nbrs_idces]
    u{j}(:,final)  = 0;
end
stDim = 2;
x_idx = b{idx}(1:stDim,1);
P_idx = zeros(stDim, stDim); % covariance matrix
% Extract columns of principal sqrt of covariance matrix
% right now we are not exploiting symmetry
for d = 1:stDim
    P_idx(:,d) = b{idx}(d*stDim+1:(d+1)*stDim, 1);
end
components_amount=2;
stDim_platf = 4;
[x_platf_comp, P_platf, w] = b2xPw(b{1}(:,1), stDim_platf, components_amount);

x_platf_weighted = zeros(2,components_amount);
for i=1:components_amount
    x_platf_weighted(:,i)=transpose(x_platf_comp{i}(3:4)*w(i));
end
x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];

cost = 0;
% u{idx}(:,final)  = 0;
% ctrlDim = size(u{idx},1);
% collision cost
cc = 0;

% State Cost
sc = 0;

% information cost
ic = 0;

% control cost
uc = 0;

rii_control = 0.1;
goal=[6;3];
Q_goal_l = 10*L*eye(stDim);
if any(final)
    delta_x = goal-x_idx;
    sc = 0.5*delta_x'*Q_goal_l*delta_x;
else
    Q_coll = 1;
    P_into_b=P_platf{1}(3:4,3:4);
    b_matrix = [x_platf;P_into_b(:)]';%b{1}()';
    for ii=2:size(b,1)
        b_matrix = [b_matrix;b{ii}'];
    end
    nSigma = sigmaToCollide_multiagent_D(D,idx,b_matrix,2,stateValidityChecker);
    for j=incoming_nbrs_idces
        cc = cc-Q_coll*log(chi2cdf(nSigma(j)^2, stDim));
    end
    
    uc = uc + rii_control*(u{idx}(:)'*u{idx}(:));
end
cost = cost + uc+sc;
% plattform_idx = 1;
% [eid,~] = inedges(D,plattform_idx);

% edge_row = idx-1;

if any(final)
    % no more rho_up term in final step
else
    sum_norm_2 = 0;
    if rho_w~=0
        for j=incoming_nbrs_idces
            sum_norm_2 = sum_norm_2 + norm(u{idx}(:) - last_com_t{j})^2;
        end

    end
    cost = cost + rho_up*lam_w(idx,:)*u{idx}(:) + rho_w*sum_norm_2;
%     u_residue = 3*u{1}(5:6,:);
%     for j = 2:4
%         u_residue = u_residue - u{j}(:,:);
%     end
%     cost = cost + rho_up/2*norm(u_residue + transpose(lam_up))^2;
end
end
