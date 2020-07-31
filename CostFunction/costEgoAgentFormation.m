function c = costEgoAgentFormation(D, idx, b, u, horizon, stDim)
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
if size(b,3)>1  
% size is n_agent * n_b * ((n_b+n_u+1)horizon) for deri before backward path
% size of u is n_agent * n_u * ((n_b+n_u+1)horizon) with NaN at the end of
% horizons


% size is n_agent * n_b for forward path
% incoming_nbrs_idces = predecessors(D,idx);
    for j=1:size(b,3)
    %     if isempty(varargin)
    %     b_this_for_paral = cell(size(b));
    %     u_this_for_paral = cell(size(u));
    %     for i = incoming_nbrs_idces
    %         b_this_for_paral{i} = b{i}(:,j);
    %         u_this_for_paral{i} = u{i}(:,j);
    %     end
    %     b_this_for_paral{idx} = b{idx}(:,j);
    %     u_this_for_paral{idx} = u{idx}(:,j);
        c(j) =  evaluateCost(D, idx, squeeze(b(:,:,j)),squeeze(u(:,:,j)), stDim, horizon);
    %     else
    %         c(i) =  evaluateCost(b(:,i),u(:,i), goal, stDim, L, stateValidityChecker, varargin{1});
    %     end
    end
end
end

function cost = evaluateCost(D, idx, b, u, stDim, L)
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
% cost = 0;
% incoming_nbrs_idces = predecessors(D,idx);

final = isnan(u(1,:));
% u(:,final)  = 0;
% for j = [idx; incoming_nbrs_idces]
%     u(j,:,final)  = 0;
% end
u(:,final)  = 0;
x_idx = transpose(b(idx,1:stDim,1));
P_idx = zeros(stDim, stDim); % covariance matrix
% Extract columns of principal sqrt of covariance matrix
% right now we are not exploiting symmetry
for d = 1:stDim
    P_idx(:,d) = b(idx,d*stDim+1:(d+1)*stDim, 1);
end
cc = 0;

% State Cost
sc = 0;

% information cost
ic = 0;

% control cost
uc = 0;
[eid,nid] = inedges(D,idx);
rij_control = 0.3;
q_formation = 1;
rii_control = 0.8;
if any(final)
    
    for j_nid = 1:length(nid)
        j = nid(j_nid);
        edge_row = eid(j_nid);
        x = transpose(b(j,1:stDim,1));
        P = zeros(stDim, stDim); % covariance matrix
    % Extract columns of principal sqrt of covariance matrix
    % right now we are not exploiting symmetry
        for d = 1:stDim
            P(:,d) = b(j,d*stDim+1:(d+1)*stDim, 1);
        end
%         RowIdx = D.Nodes.incoming_edges(idx,j);
%         ismember(D.Edges.EndNodes, [j,idx],'rows');
        formation_error = x_idx-x-(D.Edges.nom_formation_2(edge_row,:))';
        sc = sc + L*0.5*q_formation*(formation_error'*formation_error);
    end
  
else
    for j_nid = 1:length(nid)
        j = nid(j_nid);
        edge_row = eid(j_nid);
        x = transpose(b(j,1:stDim,1));
        P = zeros(stDim, stDim); % covariance matrix
    % Extract columns of principal sqrt of covariance matrix
    % right now we are not exploiting symmetry
        for d = 1:stDim
            P(:,d) = b(j,d*stDim+1:(d+1)*stDim, 1);
        end
%         RowIdx = D.Nodes.incoming_edges(idx,j);
        
%         E_j = table2array(D.Edges(RowIdx,:));
%         RowIdx = ismember(D.Edges.EndNodes, [j,idx],'rows');
        uc = uc + 0.5*rij_control*(transpose(u(j,:))'*transpose(u(j,:)));
        formation_error = x_idx-x-(D.Edges.nom_formation_2(edge_row,:))';
        sc = sc + 0.5*q_formation*(formation_error'*formation_error);
    end
    uc = uc + 0.5*rii_control*(transpose(u(idx,:))'*transpose(u(idx,:)));
end

w_cc = 1.0;
cost = sc + ic + uc + w_cc*cc;

end
