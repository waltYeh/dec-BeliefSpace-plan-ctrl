function [g,c,gb,gu,gbb,gbu,guu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
    = beliefDynCost_plattform...
    (D,idx,b,u,horizonSteps,full_DDP,motionModel,obsModel,belief_dyns)
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
%   gx: belief dynamics derivate w.r.t belief state
%   gu: belief dynamics derivate w.r.t control
%   gbb: 2-nd order derivate of belief dynamics derivate w.r.t belief state
%   gbu: belief dynamics derivate w.r.t belief state and control
%   guu: 2-nd order derivate of belief dynamics derivate w.r.t control
%   cb: cost func derivate w.r.t belief state
%   cu: cost func derivate w.r.t control
%   cbb: 2-nd order derivate of cost func derivate w.r.t belief state
%   cbu: cost func derivate w.r.t belief state and control
%   cuu: 2-nd order derivate of cost func derivate w.r.t control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
incoming_nbrs_idces = predecessors(D,idx)';
beliefDim = size(b{idx},1);
components_amount = round(beliefDim/(motionModel.stDim*(motionModel.stDim+1)+1));
% if only two outputs g and c are needed
g = cell(size(D.Nodes,1),1);
if nargout == 2
    for j = incoming_nbrs_idces
%         beliefDim = size(b{j},1);
%         size_paral = size(b{j},2);
%         g{j}=zeros(beliefDim,size_paral);
        
%         ctDim = motionModel.ctDim;% 4, only for one component
        bj=squeeze(b{j});
        uj = squeeze(u{j});
        g{j} = belief_dyns{j}(bj, uj);
    end
    g{idx} = belief_dyns{idx}(b{idx}, u{idx});
    c = cost_plattform(b{idx}, u{idx}, horizonSteps, components_amount);
else
    % belief state and control indices
    ib = 1:beliefDim;
    iu_begin = beliefDim+1;
    
    % dynamics first derivatives
    xu_dyn  = @(xu) beliefDynamicsGMM(xu(ib,:),xu(iu_begin:end,:),motionModel, obsModel);
    J       = finiteDifference(xu_dyn, [b{idx}; u{idx}]);
    gb      = J(:,ib,:);
    gu      = J(:,iu_begin:end,:);

    [gbb,gbu,guu] = deal([]);
    %% cost first derivatives
    
    xu_cost = @(xu) cost_plattform(xu(ib,:),xu(iu_begin:end,:),horizonSteps, components_amount);    
    J       = squeeze(finiteDifference(xu_cost, [b{idx}; u{idx}]));
    
    c_bi      = J(ib,:);
    c_ui      = J(iu_begin:end,:);
    
    %% cost second derivatives
    % first calculate Hessian excluding collision cost
    xu_cost_nocc = @(xu) cost_plattform(xu(ib,:),xu(iu_begin:end,:),horizonSteps, components_amount);
    xu_Jcst_nocc = @(xu) squeeze(finiteDifference(xu_cost_nocc, xu));    
    JJ      = finiteDifference(xu_Jcst_nocc, [b{idx}; u{idx}]);
    JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize                      
    
    c_bi_bi     = JJ(ib,ib,:);
    c_bi_ui     = 0.5*(JJ(ib,iu_begin:end,:)+permute(JJ(iu_begin:end,ib,:),[2 1 3]));
    c_ui_ui     = JJ(iu_begin:end,iu_begin:end,:);            
    c_ui_uj = 0;
    [g,c] = deal([]);
end
end