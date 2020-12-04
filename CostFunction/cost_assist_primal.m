function c = cost_assist_primal(D, idx, b, u,lam,rho,horizon,  ...
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
    c(j) =  evaluateCost(D, idx, b_parallel,u_parallel,...
        lam.lam_d(:,:,j),lam.lam_up(:,:,j),rho.rho_d,rho.rho_up,horizon, ...
        stateValidityChecker);
%     else
%         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, L, stateValidityChecker, varargin{1});
%     end
end

end

function cost = evaluateCost(D, idx, b, u, lam_di,lam_up,rho_d,rho_up, ...
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
if any(final)
    
else
%     nSigma = sigmaToCollide_multiagent_D(D,idx,b,2,stateValidityChecker);
%     for j=incoming_nbrs_idces
%         cc = cc-log(chi2cdf(nSigma(j)^2, stDim));
%     end
    uc = uc + rii_control*(u{idx}(:)'*u{idx}(:));
end
cost = cost + uc;
plattform_idx = 1;
[eid,~] = inedges(D,plattform_idx);

edge_row = idx-1;
formation_residue = (x_idx-x_platf-(D.Edges.nom_formation_2(edge_row,:))')*w(2)^2 ...
    +(x_idx-x_platf-(D.Edges.nom_formation_1(edge_row,:))')*w(1)^2;
cost = cost + rho_d/2*norm(formation_residue + transpose(lam_di(idx-1,:)))^2;
% consensus_residue = b{idx}(7:8,1)-x_platf;
% cost = cost + rho_d/2*norm(consensus_residue + transpose(lam_b(idx-1,:)))^2;

if any(final)
    % no more rho_up term in final step
else
    u_residue = 3*u{1}(5:6,:);
    for j = 2:4
        u_residue = u_residue - u{j}(:,:);
    end
    cost = cost + rho_up/2*norm(u_residue + transpose(lam_up))^2;
end
end
