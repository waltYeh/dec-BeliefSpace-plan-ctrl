function c = cost_plattform_primal(D, idx, b, u,...
    lam,rho,...
    horizon,stateValidityChecker)
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
        lam.lam_d(:,:,j),lam.lam_up(:,:,j),lam.lam_c(:,:,j),rho.rho_d,rho.rho_up,rho.rho_c, horizon, ...
        stateValidityChecker);
%     else
%         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, L, stateValidityChecker, varargin{1});
%     end
end

end

function cost = evaluateCost(D, idx, b, u, lam_di,lam_up,lam_c,rho_d,rho_up,rho_c, ...
    L, stateValidityChecker)
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
incoming_nbrs_idces = predecessors(D,idx)';
[eid,nid] = inedges(D,idx);
final = isnan(u{idx}(1,:));
% u(:,final)  = 0;
components_amount=2;
stDim = 4;
[x_idx, P_idx, w] = b2xPw(b{idx}(:,1), stDim, components_amount);

for j = [idx, incoming_nbrs_idces]
    u{j}(:,final)  = 0;
end
% u{idx}(:,final)  = 0;
R_t = diag([0.2, 4.0, 0.2, 0.2,0.1,0.1])*10;
Qerr_l = 100*L*eye(2);
Qerr_t = 0.0*eye(2);
Qcov_l = 10e8*eye(4); % penalize terminal covar
Qcov_l(1,1) = 0;
Qcov_l(2,2) = 0;
component_cost = zeros(components_amount,1);
x_goals = zeros(2,components_amount);
cost = 0;
for i_comp=1:components_amount
    x_goals(:,i_comp)=transpose(x_idx{i_comp}(1:2));
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
        uc = u{idx}'*R_t*u{idx};
%         sc = delta_x'*Qerr_t*delta_x;
    end
    component_cost(i_comp) = sc + ic + uc + w_cc*cc;
    
    cost = cost + component_cost(i_comp) * w(i_comp)^2;
end

x_platf_weighted = zeros(2,components_amount);
for i=1:components_amount
    x_platf_weighted(:,i)=transpose(x_idx{i}(3:4)*w(i));
end
x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];

for j_nid = 1:length(nid)-1
    j = nid(j_nid);
    edge_row = eid(j_nid);
    stDim=2;
    xj = b{j}(1:stDim,1);
% Extract columns of principal sqrt of covariance matrix
% right now we are not exploiting symmetry
    for i_comp=1:components_amount
        formation_residue = (xj-x_idx{i_comp}(3:4)-(D.Edges.nom_formation_2(edge_row,:))')*w(2)^2 ...
            +(xj-x_idx{i_comp}(3:4)-(D.Edges.nom_formation_1(edge_row,:))')*w(1)^2;
    %     rho_d = rho_d/100;
        cost = cost + rho_d/2*norm(formation_residue + transpose(lam_di(j-1,:)))^2;
    end
%     consensus_residue = b{j}(7:8,1)-x_platf;
%     cost = cost + rho_d/2*norm(consensus_residue + transpose(lam_b(j-1,:)))^2;
end

if any(final)
    % no more rho_up term in final step
    x_compl = b{5}(1:stDim,1);
    compl_residue = w(1)^2*(x_compl-x_goals(:,2))+w(2)^2*(x_compl-x_goals(:,1));
    cost = cost + rho_c/2*norm(compl_residue + transpose(lam_c))^2;

else
    u_residue = 3*u{idx}(5:6,:);
    for j_nid = 1:length(nid)-1
        j = nid(j_nid);
        u_residue = u_residue - u{j}(:,:);
    end
    cost = cost + rho_up/2*norm(u_residue + transpose(lam_up))^2;
end
end
