function [g,c]...
    =beliefDynCost_compl_primal(D,idx,b,u,lam,rho,horizonSteps,full_DDP,...
    motionModel,obsModel, belief_dyns, collisionChecker)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A utility function that combines belief dynamics and cost
% uses helper function finite_difference() to compute derivatives
% Inputs:
%   b: belief
%   u: controls
%   xf: target state
%   L: Total segments
%   full_DDP: whether to use 2nd order derivates of dynamics
%   motionModel: robot's motion model
%   obsModel: Sensing model
%   collisionChecker: collision checking with obstacles
%
% Outputs:
%   g: belief update using belief dynamics
%   c: cost 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

incoming_nbrs_idces = predecessors(D,idx)';
beliefDim = size(b{idx},1);
ctrlDim = size(u{idx},1);
paralDim = size(b{idx},2);% which can be one or equal to horizonSteps or 11

% only for debug
if paralDim == 11
    a=1;
end
% neighbors_amount = round(beliefDim/(motionModel.stDim*(motionModel.stDim+1)));
% ctDim = motionModel.ctDim;% 4, only for one component
% if only two outputs g and c are needed
g = cell(size(D.Nodes,1),1);
% b_formation = zeros(size(D.Nodes,1),beliefDim,paralDim);
% u_formation = zeros(size(D.Nodes,1),ctrlDim,paralDim);
% 
% for j = incoming_nbrs_idces
%     % the belief of agent idx about agent i
%     %the last ":" in the following is for the parallel computing, not
%     %affecting single computation
%     bj=squeeze(b{j});
%     uj = squeeze(u{j});
%     
%     g{j} = belief_dyns{j}(bj, uj);
%     b_formation(j,:,:) = bj;
%     u_formation(j,:,:) = uj;
%     
% end
% b_formation(idx,:,:) = b{idx};
% u_formation(idx,:,:) = u{idx};

g{idx} = belief_dyns{idx}(b{idx}, u{idx});
%     c = costAssistingRobot(b{idx}, u{idx}, horizonSteps, motionModel.stDim,components_amount);

c = cost_compl_primal(D, idx,b, u, ...
    lam,rho,...
    horizonSteps, collisionChecker);
end
