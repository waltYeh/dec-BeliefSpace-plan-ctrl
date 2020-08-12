function [g,c,gb,gu,gbb,gbu,guu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
    = beliefDynCost_crane(D,idx,b,u,horizonSteps,full_DDP,motionModel,obsModel,belief_dyns, collisionChecker)
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
b_formation = zeros(size(D.Nodes,1),beliefDim,paralDim);
u_formation = zeros(size(D.Nodes,1),ctrlDim,paralDim);
if nargout == 2
    for j = incoming_nbrs_idces
        % the belief of agent idx about agent i
        %the last ":" in the following is for the parallel computing, not
        %affecting single computation
        bj=squeeze(b{j});
        uj = squeeze(u{j});
        g{j} = belief_dyns{j}(bj, uj);
        if size(bj,1)>6
            for k=1:paralDim
                [x, P, w] = b2xPw(bj(:,k), 4, 2);
                avr_x = w(1)*x{1}(3:4,:)+w(2)*x{2}(3:4,:);
                avr_P = w(1)*P{1}(3:4,3:4)+w(2)*P{2}(3:4,3:4);
                b_formation(j,:,k) = [avr_x;avr_P(:)];
                
            end
%             [x, P, w] = b2xPw(bj, 4, 2);
%             avr_x = w(1)*x{1}(3:4,:)+w(2)*x{2}(3:4,:);
%             avr_P = w(1)*P{1}(3:4,3:4)+w(2)*P{2}(3:4,3:4);
%             b_formation(j,:,:) = [avr_x;avr_P(:)];
            u_formation(j,:,:) = uj(5:6,:);
        else
            b_formation(j,:,:) = bj;
            u_formation(j,:,:) = uj;
        end
    end
    b_formation(idx,:,:) = b{idx};
    u_formation(idx,:,:) = u{idx};
    g{idx} = belief_dyns{idx}(b{idx}, u{idx});
%     c = costAssistingRobot(b{idx}, u{idx}, horizonSteps, motionModel.stDim,components_amount);
    
    c = costAgentFormation(D, idx,b_formation, u_formation, horizonSteps, motionModel.stDim, collisionChecker);
else
    % belief state and control indices
    ib = 1:beliefDim;
    iu_begin = beliefDim+1;
    
    % dynamics first derivatives, dynamics is decoupled!, there is no
    % g^i_uj, only g^i_ui
    xu_dyn  = @(xu) beliefDynamicsSimpleAgent(xu(ib,:),xu(iu_begin:end,:),motionModel, obsModel);
%     J=cell(size(D.Nodes,1),1);
%     gb=cell(1,size(D.Nodes,1));
%     gu=cell(1,size(D.Nodes,1));
%     cb=cell(1,size(D.Nodes,1));
%     cu=cell(1,size(D.Nodes,1));
    
%     for j = incoming_nbrs_idces
%         J{j}       = finiteDifference(xu_dyn, [b{j}; u{j}]);
%         gb{idx,j}      = J(:,ib,:);
%         gu{idx,j}      = J(:,iu_begin:end,:);
%     end

% only deal with idx agent, but not any other agents
% but for cost values without der, b and u of all neighboring agents are
% passed in, now we dont need those neighbors, but still have to pass them
% in so that the syntax can be consistent
    J       = finiteDifference(xu_dyn, [b{idx}; u{idx}]);
    gb      = J(:,ib,:);
    gu      = J(:,iu_begin:end,:);
    % gu depends on dt
%     all others are zero because dyn is decoupled
    % dynamics second derivatives
%     if full_DDP
%         xu_Jcst = @(xu) finiteDifference(xu_dyn, xu);
%         JJ      = finiteDifference(xu_Jcst, [b; u]);
%         JJ      = reshape(JJ, [4 6 size(J)]);
%         JJ      = 0.5*(JJ + permute(JJ,[1 3 2 4])); %symmetrize
%         gbb     = JJ(:,ib,ib,:);
%         gbu     = JJ(:,ib,iu,:);
%         guu     = JJ(:,iu,iu,:);
%     else
        [gbb,gbu,guu] = deal([]);
%     end
    
%      if motionModel.stDim ~= 2
%         error('This partial of f w.r.t sigma is only valid for robot with state dimension 2')
%      end
%     
%     %% First derivative of sigmaToCollide (jacobian of sigma[b])
%     tStart = tic;
%     xu_sigma =  @(x) sigmaToCollide(x, motionModel.stDim, collisionChecker);
%     dsigma_db  = squeeze(finiteDifference(xu_sigma, b,1e-1)); % need to have a large step size to see derivative in collision
%     dsigma_db = [dsigma_db;zeros(motionModel.ctDim,size(dsigma_db,2))]; % jacobian w.r.t u is zero for collision
%     
%     nSigma = sigmaToCollide(b, motionModel.stDim, collisionChecker);
%     fprintf('Time to do sigma derivative and compute sigma: %f seconds\n', toc(tStart))
%     
    %% cost first derivatives, only for u_i is enough
    for j = incoming_nbrs_idces
        % the belief of agent idx about agent i
        %the last ":" in the following is for the parallel computing, not
        %affecting single computation
        bj=squeeze(b{j});
        uj = squeeze(u{j});
        if size(bj,1)>6
            for k=1:horizonSteps
                [x, P, w] = b2xPw(bj(:,k), 4, 2);
                avr_x = w(1)*x{1}(3:4,:)+w(2)*x{2}(3:4,:);
                avr_P = w(1)*P{1}(3:4,3:4)+w(2)*P{2}(3:4,3:4);
                b_formation(j,:,k) = [avr_x;avr_P(:)];
                
            end
            u_formation(j,:,:) = uj(5:6,:);
        else
            b_formation(j,:,:) = bj;
            u_formation(j,:,:) = uj;
        end
    end
    b_formation(idx,:,:) = b{idx};
    u_formation(idx,:,:) = u{idx};
    xu_cost = @(xu) costAgentFormation(D,idx,xu(:,ib,:),xu(:,iu_begin:end,:),horizonSteps,motionModel.stDim, collisionChecker);    
%     J       = 
    
%     % construct Jacobian adding collision cost
%     for i = 1:size(dsigma_db,2)               
%         J(:,i) = J(:,i) + ((-1/2)/(exp(nSigma(i)/2)-1)) * dsigma_db(:,i);
%     end
    
%     cb      = J(ib,:);
%     cu      = J(iu,:);
%     cb=zeros(size(D.Nodes,1),beliefDim,horizon);
%     cu=zeros(size(D.Nodes,1),ctrlDim,horizon);
%     for j = incoming_nbrs_idces
% input dim 4*8*41
        J       = multiAgentFiniteDifference(xu_cost,D,idx, squeeze(cat(2,b_formation(:,:,:), u_formation(:,:,:))));
        
        xu_sigma =  @(b_f) sigmaToCollide_multiagent_D(D,idx,b_f(:,:,:),motionModel.stDim,collisionChecker);
        
        
        dsigma_db  = squeeze(multiAgentFiniteDifference(xu_sigma,D,idx, b_formation,1e-1)); % need to have a large step size to see derivative in collision
        % J has the size 4x8x61, dsigma_db 4x6x61 -> 4x8x61
        dsigma_db = cat(2,dsigma_db,zeros(size(D.Nodes,1),motionModel.ctDim,size(dsigma_db,3))); % jacobian w.r.t u is zero for collision
        nSigma = sigmaToCollide_multiagent_D(D,idx,b_formation, motionModel.stDim, collisionChecker);
%         bb =1;
        for j = [idx,incoming_nbrs_idces]
            for k = 1:size(dsigma_db,3)  %horizon             
                J(j,:,k) = J(j,:,k) + ((-1/2)/(exp(nSigma(j,k)/2)-1)) * dsigma_db(j,:,k);
                
            end
%             bb=bb+1;
        end
        
        c_bi      = squeeze(J(idx,ib,:));% 1x6x41
        c_ui      = squeeze(J(idx,iu_begin:end,:));%1x2x41
%     end
%     J       = finiteDifference(xu_cost, squeeze(cat(2,b(idx,:,:), u(idx,:,:))));
%     cb(idx,:,:)     = J(1,ib,:);
%     cu(idx,:,:)      = J(1,iu_begin:end,:);
    %% cost second derivatives
    
    
    % first calculate Hessian excluding collision cost
    xu_cost_nocc = @(xu) costAgentFormation(D,idx,xu(:,ib,:),xu(:,iu_begin:end,:),horizonSteps,motionModel.stDim, collisionChecker);
    xu_Jcst_nocc = @(xu) squeeze(multiAgentFiniteDifference(xu_cost_nocc,D,idx, xu));   
    % the following can only compute c_uj_uj
    % JJ = finiteDifference(fun, x, h)
%     for j = incoming_nbrs_idces
%         JJ{j}   = finiteDifference(xu_Jcst_nocc, squeeze(cat(2,b(:,:,:), u(:,:,:))));
%         JJ{j}     = 0.5*(JJ{j} + permute(JJ{j},[2 1 3]));%symmetrize  
%         cbb{j}     = JJ{j}(ib,ib,:);
%         cbu{j}     = JJ{j}(ib,iu_begin:end,:);
%         cuu{j}     = JJ{j}(iu_begin:end,iu_begin:end,:); 
%     end
% input dim 4*8*41
    JJ   = multiAgentFiniteDifference2(xu_Jcst_nocc, D,idx,squeeze(cat(2,b_formation(:,:,:), u_formation(:,:,:))));
    % construct Hessian adding collision cost
%     for i = 1:size(dsigma_db,2)
%         jjt = dsigma_db(:,i)*dsigma_db(:,i)';        
%         JJ(:,:,i) = JJ(:,:,i) + ((1/4)*exp(nSigma(i)/2)/(exp(nSigma(i)/2)-1)^2) * 0.5*(jjt+jjt');
%     end
    %4x8x8x41
    %     JJ{idx}      = 0.5*(JJ{idx} + permute(JJ{idx},[2 1 3]));%symmetrize  
%     cbb{idx}     = JJ{idx}(ib,ib,:);
%     cbu{idx}     = JJ{idx}(ib,iu_begin:end,:);
%     cuu{idx}     = JJ{idx}(iu_begin:end,iu_begin:end,:); 
    % construct Hessian adding collision cost
    for j = [idx,incoming_nbrs_idces]
        for k = 1:size(dsigma_db,3)
            jjt = dsigma_db(j,:,k)*dsigma_db(j,:,k)';        
            JJ(j,:,:,k) = JJ(j,:,:,k) + ((1/4)*exp(nSigma(j,k)/2)/(exp(nSigma(j,k)/2)-1)^2) * 0.5*(jjt+jjt');
        end
    end 
    
    c_bi_bi = squeeze(JJ(idx,ib,ib,:));% the more edges coming into i, the higher
    % usually eye or 2*eye
    c_bi_ui = squeeze(0.5 * (JJ(idx,ib,iu_begin:end,:) + permute(JJ(idx,iu_begin:end,ib,:),[1 3 2 4])));
    c_ui_ui = squeeze(JJ(idx,iu_begin:end,iu_begin:end,:));
    % the answer is fixed diag(rii_control,rii_control) for the horizon
    c_ui_uj = zeros(size(D.Nodes,1),ctrlDim,ctrlDim,horizonSteps);
    c_ui_uj(incoming_nbrs_idces,:,:,:) = JJ(incoming_nbrs_idces,iu_begin:end,iu_begin:end,:);
%     JJ      = finiteDifference(xu_Jcst_nocc, [b; u]);
%     JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize                      
%     
   
%     cbb     = JJ(ib,ib,:);
%     cbu     = JJ(ib,iu,:);
%     cuu     = JJ(iu,iu,:);            

    [g,c] = deal([]);
end
end
