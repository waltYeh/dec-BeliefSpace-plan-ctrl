function [x,cost_new] = iLQG_rollout(D,idx,DYNCST, x0, u, u_lims)

        [x,un,cost_new]  = forward_pass(D,idx,x0,u,1,DYNCST, u_lims);



% if ~isempty(lambda_last)
%     figure(52)
%     subplot(2,2,idx):
%     plot([iter-1,iter],[lambda_last,lambda])
%     hold on
%     figure(53)
%     subplot(2,2,idx)
%     plot([iter-1,iter],[dlambda_last,dlambda])
%     hold on
% end
% update trace
% trace(iter).lambda      = lambda;
% trace(iter).dlambda     = dlambda;
% trace(iter).alpha       = alpha;
% trace(iter).improvement = dcost;
% trace(iter).cost        = sum(cost(:));
% trace(iter).reduc_ratio = z;
%     stop = graphics(Op.plot,x,u,cost,L,Vx,Vxx,fx,fxx,fu,fuu,trace(1:iter),0);



% [x,un,cost]  = forward_pass(x0(:,1),alpha*u,[],[],[],1,DYNCST,Op.lims,[]);
function [xnew,unew,cnew] = forward_pass(D,idx,x0,u,Alpha,DYNCST,lims)
% l (Schwarting j_k^i) is taken into the function as the argument du
% parallel forward-pass (rollout)
% internally time is on the 3rd dimension, 
% to facillitate vectorized dynamics calls
% Alpha is a series of possible step length, 
% it is multiplied with du(:,i) in parallel
n_agent = size(D.Nodes,1);
% n_b        = size(x0,2);
K        = length(Alpha);
K1       = ones(1,K); % useful for expansion

horizon        = size(u{1},2);
xnew = cell(n_agent,1);
unew = cell(n_agent,1);
% uC_lambda_new = cell(n_agent,1);
for i=1:n_agent
    n_b = size(x0{i},1);
    m_u = size(u{i},1);
    xnew{i} = zeros(n_b,K,horizon+1);
    xnew{i}(:,:,1) = x0{i}(:,ones(1,K));
    unew{i} = zeros(m_u,K,horizon);
end
% Dim_lam_in_xy = 2;
% lam_d_new = zeros(n_agent-1,Dim_lam_in_xy,K,horizon+1);
% lam_up_new = zeros(1,Dim_lam_in_xy,K,horizon);
cnew        = zeros(1,1,K,horizon+1);% one agent, one dim c
for k = 1:horizon
    for i=1:n_agent
        unew{i}(:,:,k) = u{i}(:,k*K1);
    end
    if ~isempty(lims)
        % add u upper and lower bound again
%         for i=1:size(D.Nodes,1)
            unew_idxk = squeeze(unew{idx}(:,:,k));
            size_unew = size(unew_idxk);
            m_u = size(u{idx},1);
            if size_unew(2)==m_u
                unew_ik = transpose(unew_ik);
            end
            unew{idx}(:,:,k) = min(lims(:,2*K1), max(lims(:,1*K1), unew_idxk));
%         end
    end

% unew of other agents is updated, causing change of formation cost, which vergiftet the cost
    xnew_k = cell(n_agent,1);
    unew_k = cell(n_agent,1);
%     lam_d_k = zeros(n_agent-1,Dim_lam_in_xy,K);
%     lam_up_k = zeros(1,Dim_lam_in_xy,K);
    for i=1:n_agent
        xnew_k{i}=xnew{i}(:,:,k);
        unew_k{i}=unew{i}(:,:,k);
    end
    [xnew_next,cnew_k]  = DYNCST(D,idx,xnew_k, unew_k, k*K1);
    incoming_nbrs_idces = predecessors(D,idx)';
    for i=1:n_agent
        if ~isempty(xnew_next{i})
            xnew{i}(:,:,k+1) = xnew_next{i};
        end
    end
    xnew{idx}(:,:,k+1) = xnew_next{idx};
    cnew(:,:,:,k)=cnew_k;
end
xnew_k = cell(n_agent,1);
unew_k = cell(n_agent,1);
for i=1:n_agent
    xnew_k{i}=xnew{i}(:,:,horizon+1);
    unew_k{i}=nan(size(unew{i},1),K,1);
end
[~,cnew(:,:,:,horizon+1)]  = DYNCST(D,idx,xnew_k, unew_k, k*K1);
% [~, cnew(:,:,:,horizon+1)] = DYNCST_primal(D,idx,xnew_k,unew_k,lam_d_k,lam_up_k,k);
for i=1:n_agent
    xnew{i} = permute(xnew{i}, [1 3 2 ]);
    unew{i} = permute(unew{i}, [1 3 2 ]);%unew of other agents never updated
end
cnew = permute(cnew, [1 2 4 3 ]);
