function [x, u,L, Vx, Vxx, cost,  ...
    lambda, dlambda, finished,flgChange,derivatives_cell] ...
    = iLQG_admm_one_iter(D,idx,DYNCST,DYNCST_primal,DYNCST_primal_diff, x0, Op, iter,...
    u_guess,lam_di,lam_up,lambda_last, dlambda_last, ...
    u_last, x_last, cost_last,...
    flg_last, derivatives_cell_last, u_lims)

defaults = {'lims',           [],...            control limits
            'parallel',       true,...          use parallel line-search?
            'Alpha',          10.^linspace(0,-3,11),... backtracking coefficients
            'tolFun',         1e-1,...          reduction exit criterion
            'tolGrad',        1e-4,...          gradient exit criterion
            'maxIter',        50,...           maximum iterations            
            'lambda',         1,...             initial value for lambda
            'dlambda',        1,...             initial value for dlambda
            'rho',            [1,1],...        %too large, converge slow,  rho_x and rho_u
            'lambdaFactor',   1.4,...           lambda scaling factor
            'lambdaMax',      1e5,...          lambda maximum value
            'lambdaMin',      1e-6,...          below this value lambda = 0
            'regType',        1,...             regularization type 1: q_uu+lambda*eye(); 2: V_xx+lambda*eye()
            'zMin',           0,...             minimal accepted reduction ratio
            'diffFn',         [],...            user-defined diff for sub-space optimization
            'plot',           0,...             0: no;  k>0: every k iters; k<0: every k iters, with derivs window
            'print',          3,...             0: no;  1: final; 2: iter; 3: iter, detailed
            'plotFn',         @(x)0,...         user-defined graphics callback
            'cost',           [],...            initial cost for pre-rolled trajectory            
            };
% x0: {1x4},42x1, u: {1x4},6x60,2x60,...
        
n   = size(x0, 2);          % dimension of belief state vector
m   = size(u_guess, 2);          % dimension of control vector
N   = size(u_guess, 3);          % number of state transitions time steps

Op  = setOpts(defaults,Op);
verbosity = Op.print;

finished = false;
derivatives_cell = {};
if iter == 1
    u = u_guess;
%     uC_lambda = uC_lambda_guess;
%     switch numel(Op.lims)
%     case 0
%     case 2*m
%         % we are here, no change in fact
%         Op.lims = sort(Op.lims,2);
%     case 2
%         Op.lims = ones(m,1)*sort(Op.lims(:))';
%     case m
%         Op.lims = Op.lims(:)*[-1 1];
%     otherwise
%         error('limits are of the wrong size')
%     end
    lambda   = Op.lambda;
    dlambda  = Op.dlambda;
    for alpha = Op.Alpha
        % x, only nodes pointing towards me and myself is filled with
        % belief state predictions
        alpha_u = u;
        for i=1:length(alpha_u)
            alpha_u{i} = alpha*u{i};
        end
        [x,un,cost,cost_origin]  = forward_pass(D,idx,x0,alpha_u,[],[],[],[],1,DYNCST,...
            DYNCST_primal,u_lims,[],lam_di,lam_up);
        incoming_nbrs_idces = predecessors(D,idx)';
        diverges = false;
        for j = [idx,incoming_nbrs_idces]
            if all(abs(x{j}(:)) < 1e8)
%                 u = un;
%                 break
            else
                diverges = true;
            end
            
        end
        if ~diverges
            u = un;
            break
        end
    end
%     trace(1).cost = sum(cost(:));
    flgChange   = 1;
else
    cost = cost_last;
    x = x_last;
    u = u_last;
%     uC_lambda = uC_lambda_last;
    lambda = lambda_last; 
    dlambda = dlambda_last;
    flgChange = flg_last;
end
  
%====== STEP 1: differentiate dynamics and cost along new trajectory
% if flgChange
    enlonged_u = u;
%     enlonged_uC_lambda = uC_lambda;
    if size(u{1},2)<size(x{1},2)
        for i=1:size(D.Nodes,1)
            ctrl_dim = size(u{i},1);
            enlonged_u{i}=cat(2,u{i},nan(ctrl_dim,1));
    %         enlonged_uC_lambda{i}=cat(2,uC_lambda{i},nan(ctrl_dim,1));
        end
    else
        
    end
    % only considers agent idx itself fx: 6x6x41, fu 6x2x41, c_bi 6x41, cui
    % 2x41, ... 6x6x41, 6x2x41, 2x2x41, c_ui_uj 4x2x2x41
    [~,~,fx,fu,fxx,fxu,fuu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = DYNCST(D,idx,x, enlonged_u, 1:N+1);
    [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = DYNCST_primal_diff(D,idx,x,enlonged_u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam_di,lam_up);

%     flgChange   = 0;
% else
% %     fx,fu,fxx,fxu,fuu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj = derivatives_cell_last{1};
%     fx = derivatives_cell_last{1};
%     fu = derivatives_cell_last{2};
%     fxx = derivatives_cell_last{3};
%     fxu = derivatives_cell_last{4};
%     fuu = derivatives_cell_last{5};
%     c_bi = derivatives_cell_last{6};
%     c_ui = derivatives_cell_last{7};
%     c_bi_bi = derivatives_cell_last{8};
%     c_bi_ui = derivatives_cell_last{9};
%     c_ui_ui = derivatives_cell_last{10};
%     c_ui_uj = derivatives_cell_last{11};
% end
% convergence or not, these plots are the same!
% figure(1+50)
% subplot(2,2,idx)
% horizonSteps = size(x,3);
% plot(1:horizonSteps,squeeze(c_bi(1,:)),'b.')
% hold on
% plot(1:horizonSteps,squeeze(c_bi(2,:)),'r')
% hold on
% title(strcat('c_bi of agent ',num2str(idx)))
% figure(1+100)
% subplot(2,2,idx)
% plot(1:horizonSteps,squeeze(c_ui(1,:)),'b.')
% hold on
% plot(1:horizonSteps,squeeze(c_ui(2,:)),'r')
% hold on
% title(strcat('c_ui of agent ',num2str(idx)))

%====== STEP 2: backward pass, compute optimal control law and cost-to-go
backPassDone   = 0;
while ~backPassDone
    [diverge, Vx, Vxx, l, L, dV, Ku] = back_pass(D,idx,c_bi,c_ui,c_bi_bi,...
        c_bi_ui,c_ui_ui,c_ui_uj,fx,fu,fxx,fxu,fuu,lambda,Op.regType,u_lims,u);
    % l is the feedforward term (42), 
    % L is the time-variant feedback(42)
%     trace(iter).time_backward = toc(t_back);
% figure(2+50)
% subplot(2,2,idx)
% horizonSteps = size(x,3);
% plot(1:horizonSteps-1,squeeze(l(1,:)),'b.')
% hold on
% plot(1:horizonSteps-1,squeeze(l(2,:)),'r')
% hold on
% title(strcat('l of agent ',num2str(idx)))
% figure(2+100)
% subplot(2,2,idx)
% plot(1:horizonSteps-1,squeeze(L(1,1,:)),'b.')
% hold on
% plot(1:horizonSteps-1,squeeze(L(2,2,:)),'r')
% hold on
% title(strcat('Lx of agent ',num2str(idx)))

    if diverge
        % maybe not step 13 in Algorithm

        fprintf('Cholesky failed at timestep %d.\n',diverge);

        dlambda   = max(dlambda * Op.lambdaFactor, Op.lambdaFactor);
        lambda    = max(lambda * dlambda, Op.lambdaMin);
        if lambda > Op.lambdaMax
            break;
        end
        continue
    end
    backPassDone      = 1;
end

% check for termination due to small gradient
% another exit criteria than Schwarting, but it hardly ever comes here
g_norm         = mean(max(abs(l) ./ (squeeze(abs(u{idx}(:,:)+1))),[],1));
% trace(iter).grad_norm = g_norm;
if g_norm < Op.tolGrad && lambda < 1e-5
    dlambda   = min(dlambda / Op.lambdaFactor, 1/Op.lambdaFactor);
    lambda    = lambda * dlambda * (lambda > Op.lambdaMin);
    if verbosity > 0
        fprintf('\nSUCCESS: gradient norm < tolGrad\n');
    end
    finished = true;
    return;
end

%====== STEP 3: line-search to find new control sequence, trajectory, cost
% [~,~,cost]  = forward_pass(D,idx,x0,u,[],[],[],[],1,DYNCST,u_lims,[]);
fwdPassDone  = 0;
if backPassDone
    if Op.parallel  % parallel line-search
        %only u is different for case of consensus and direct exchange
        Op.Alpha(end)=0;
        [xnew,unew,costnew,costnew_origin] = forward_pass(D,idx,x0 ,u, L, x, l, Ku,...
            Op.Alpha, DYNCST,DYNCST_primal,u_lims,Op.diffFn,lam_di,lam_up);

        if iter>1
            figure(4+50)
            subplot(2,2,idx)
            horizonSteps = size(x{idx},2);
            plot(1:horizonSteps,squeeze(costnew_origin(:,:,:,1)),'-')
            hold on
    %         plot(1:horizonSteps,squeeze(costnew(:,:,:,4)))
    %         plot(1:horizonSteps,squeeze(costnew(:,:,:,8)))

            % plot(1:horizonSteps-1,squeeze(l(2,:)),'r')
            % hold on
            title(strcat('cost of agent ',num2str(idx)))
        end
% figure(2+100)
% subplot(2,2,idx)
% plot(1:horizonSteps-1,squeeze(L(1,1,:)),'b.')
% hold on
% plot(1:horizonSteps-1,squeeze(L(2,2,:)),'r')
% hold on
% title(strcat('Lx of agent ',num2str(idx)))
%         
        % now we have 10 candidates of new traj
        Dcost               = sum(squeeze(costnew(:,:,:,end))) - sum(squeeze(costnew),1);
        Dcost_origin = sum(squeeze(costnew_origin(:,:,:,end))) - sum(squeeze(costnew_origin),1);
        [dcost, w]          = max(Dcost);
        dcost_origin = Dcost_origin(w);
        % find the one with greatest cost reduction
        alpha               = Op.Alpha(w);
        expected            = -alpha*(dV(1) + alpha*dV(2));
        if expected > 0
            z = dcost/expected;
        else
            z = sign(dcost);
            warning('non-positive expected reduction');
        end
        if (z > Op.zMin)||w<8
            
            fwdPassDone = 1;
            costnew     = costnew(:,:,:,w);
            for i=1:size(D.Nodes,1)
                xnew{i}        = xnew{i}(:,:,w);
                unew{i}        = unew{i}(:,:,w);
            end
        else%still move a tiny step to avoid stucking in local places
            
            fwdPassDone = 0;
            costnew     = costnew(:,:,:,w);
            for i=1:size(D.Nodes,1)
                xnew{i}        = xnew{i}(:,:,w);
                unew{i}        = unew{i}(:,:,w);
            end
        end
        costnew_origin     = costnew_origin(:,:,:,w);
    end
    if ~fwdPassDone
%         alpha = nan; % signals failure of forward pass
    end
%     trace(iter).time_forward = toc(t_fwd);
end

%====== STEP 4: accept step (or not), draw graphics, print status

% print headings
if verbosity > 1
    if idx==1
        fprintf('\n');
        fprintf('%-12s','idx','iteration','cost','reduction','alpha','expected','gradient','lambda');
        fprintf('\n');
    end
end

if fwdPassDone
    derivatives_cell =  { fx,fu,fxx,fxu,fuu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj};

    
    % maybe step 13 in Algorithm
    % decrease lambda
    dlambda   = min(dlambda / Op.lambdaFactor, 1/Op.lambdaFactor);
    lambda    = lambda * dlambda * (lambda > Op.lambdaMin);

    % accept changes
    u              = unew;
    x              = xnew;
    cost           = costnew;
    flgChange      = 1;
%         drawResult(Op.plotFn,x,2);
%         Op.plotFn(x);

    % terminate ?
%     dcost_origin
    % print status
    if verbosity > 1
        fprintf('\n%-12d%-12d%-12.3g%-12.3g%-12.3g%-12d%-12.3g%-12.3g%-12.3g\n', ...
           idx, iter, sum(costnew_origin(:)), dcost_origin,alpha,expected, g_norm, lambda);
    end
    if dcost_origin < Op.tolFun && dcost_origin>0 && iter > 5
        if verbosity > 0
            fprintf('\nSUCCESS: cost change < tolFun\n');
        end
        finished = true;
        return;
    end

else % no cost improvement
    
    derivatives_cell =  { fx,fu,fxx,fxu,fuu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj};
    % increase lambda
    dlambda  = max(dlambda * Op.lambdaFactor, Op.lambdaFactor);
    lambda   = max(lambda * dlambda, Op.lambdaMin);
%     dlambda   = min(dlambda / Op.lambdaFactor, 1/Op.lambdaFactor);
%     lambda    = lambda * dlambda * (lambda > Op.lambdaMin);
    u              = unew;
    x              = xnew;
    cost           = costnew;
    flgChange      = 1;% do the der again, because the consensus will help improve
    % print status
    if verbosity > 1
        fprintf('\n%-12d%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.3g%-12.3g\n', ...
            idx, iter,sum(costnew_origin(:)), dcost_origin, alpha,expected, g_norm, lambda);           
    end     

    % terminate ?
    if lambda > Op.lambdaMax
        if verbosity > 0
            fprintf('\nEXIT: lambda > lambdaMax\n');
        end
        finished = true;
        return;
    end
end
figure(31)
subplot(2,2,idx)
plot(iter,sum(cost(:)),'.')
hold on

% if ~isempty(lambda_last)
%     figure(52)
%     subplot(2,2,idx)
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
function [xnew,unew,cnew,cnew_origin] = forward_pass(D,idx,x0,u,L,x,l,Ku,Alpha,DYNCST,...
    DYNCST_primal,lims,diff,lam_d,lam_up)
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
Dim_lam_in_xy = 2;
lam_d_new = zeros(n_agent-1,Dim_lam_in_xy,K,horizon+1);
lam_up_new = zeros(1,Dim_lam_in_xy,K,horizon);
cnew        = zeros(1,1,K,horizon+1);% one agent, one dim c
cnew_origin = zeros(1,1,K,horizon+1);
for k = 1:horizon
    for i=1:n_agent
        unew{i}(:,:,k) = u{i}(:,k*K1);
%         uC_lambda_new{i}(:,:,k) = uC_lambda{i}(:,k*K1);
    end
    for i=1:n_agent-1
        lam_d_new(i,:,:,k) = lam_d(i,:,k*K1);
%         uC_lambda_new{i}(:,:,k) = uC_lambda{i}(:,k*K1);
    end
    lam_up_new(1,:,:,k) = lam_up(1,:,k*K1);
    if ~isempty(l)
        % feedforward control term should not be too agressive, Alpha < 1
        unew{idx}(:,:,k) = squeeze(unew{idx}(:,:,k)) + l(:,k)*Alpha;
    end    
    
    if ~isempty(L)%!!! not corrected yet
        if ~isempty(diff)
            dx = diff(xnew(:,:,k), x(:,k*K1));
        else
            dx          = xnew{idx}(:,:,k) - x{idx}(:,k*K1);% both size 1x6x11
        end
        unew{idx}(:,:,k) = squeeze(unew{idx}(:,:,k)) + L(:,:,k)*squeeze(dx); 
        % with feedback
    end%!!! not corrected yet
    if ~isempty(Ku)%!!! not corrected yet
        du_all_agent = u;
%         u_real_all_agent = u;
%         du_all_agent(:) = 0;
%         dx          = xnew(idx,:,:,k) - x(idx,:,k*K1);
        incoming_nbrs = predecessors(D,idx)';
%         for j_agent=incoming_nbrs
%             unew(idx,:,:,k) = squeeze(unew(idx,:,:,k)) + squeeze(Ku(j_agent,:,:,k))'*transpose(squeeze(du_all_agent(j_agent,:,k)));
%         end
         % with feedback
    end%!!! not corrected yet
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
    lam_d_k = zeros(n_agent-1,Dim_lam_in_xy,K);
    lam_up_k = zeros(1,Dim_lam_in_xy,K);
    for i=1:n_agent
        xnew_k{i}=xnew{i}(:,:,k);
        unew_k{i}=unew{i}(:,:,k);
        
%         u_lambda_k = u{i}(:,:,k);
    end
    for i=1:n_agent-1
        lam_d_k(i,:,:) = lam_d_new(i,:,:,k);
    end
    lam_up_k(1,:,:) = lam_up_new(1,:,:,k);
    [xnew_next,cnew_origin_k]  = DYNCST(D,idx,xnew_k, unew_k, k*K1);
    [~,cnew_k]=DYNCST_primal(D,idx,xnew_k, unew_k, lam_d_k,lam_up_k ,k*K1);
    incoming_nbrs_idces = predecessors(D,idx)';
    for i=1:n_agent
        xnew{i}(:,:,k+1) = xnew_next{i};
        
    end
    xnew{idx}(:,:,k+1) = xnew_next{idx};
    cnew(:,:,:,k)=cnew_k;
    cnew_origin(:,:,:,k)=cnew_origin_k;
end
xnew_k = cell(n_agent,1);
    unew_k = cell(n_agent,1);
    for i=1:n_agent
        xnew_k{i}=xnew{i}(:,:,horizon+1);
        unew_k{i}=nan(size(unew{i},1),K,1);
%         lam_d_k{i} = nan(size(unew{i},1),K,1);
    end
    for i=1:n_agent-1
        lam_d_k(i,:,:) = lam_d_new(i,:,:,horizon+1);
    end
    lam_up_k(1,:,:) = nan(Dim_lam_in_xy,K,1);
    [~, cnew_origin(:,:,:,horizon+1)] = DYNCST(D,idx,xnew_k,unew_k,k);
    [~, cnew(:,:,:,horizon+1)] = DYNCST_primal(D,idx,xnew_k,unew_k,lam_d_k,lam_up_k,k);
for i=1:n_agent
    xnew{i} = permute(xnew{i}, [1 3 2 ]);
    unew{i} = permute(unew{i}, [1 3 2 ]);%unew of other agents never updated
end
cnew = permute(cnew, [1 2 4 3 ]);
cnew_origin = permute(cnew_origin, [1 2 4 3 ]);
% components_amount = 2;
% horiz = N+1;
% mu = cell(components_amount,1);
% sig = cell(components_amount,1);
% weight = zeros(components_amount,horiz);
% for i_comp = 1:components_amount
%     mu{i_comp}=zeros(4,horiz);
%     sig{i_comp}=zeros(16,horiz);
% %     weight(i_comp,i)=w(i_comp);
% end
% for i=1:horiz
%     [x, P, w] = b2xPw(xnew(:,:,i), 4, components_amount);
%     for i_comp = 1:components_amount
%         mu{i_comp}(:,i)=x{i_comp};
%         sig{i_comp}(:,i)=P{i_comp}(:);
%         weight(i_comp,i)=w(i_comp);
%     end
% end
% figure(11)
% hold on
% for i_comp = 1:components_amount
%     plot(mu{i_comp}(1,:),mu{i_comp}(2,:),'.')
%     axis equal
%     plot(mu{i_comp}(3,:),mu{i_comp}(4,:),'+')
% end
% 
% for k=1:horiz
%     pointsToPlot = drawResultGMM([mu{1}(:,k); sig{1}(:,k)], 4);
%     figure(8)
%     plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
%     pointsToPlot = drawResultGMM([mu{2}(:,k); sig{2}(:,k)], 4);
%     plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
%     pause(0.1);
% end
% hold off

% put the time dimension in the columns


function [diverge, Vx, Vxx, k, K, dV, Ku] = back_pass(D,idx,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,fx,fu,fxx,fxu,fuu,lambda,regType,lims,u)
% Perform the Ricatti-Mayne backward pass
% cx=0;cu=0;cxx=0;cxu=0;cuu=0;
% tensor multiplication for DDP terms
vectens = @(a,b) permute(sum(bsxfun(@times,a,b),1), [3 2 1]);

N  = size(c_bi,2);
n  = numel(c_bi)/N;
m  = numel(c_ui)/N;

cx    = reshape(c_bi,  [n N]);
cu    = reshape(c_ui,  [m N]);
cxx   = reshape(c_bi_bi, [n n N]);
cxu   = reshape(c_bi_ui, [n m N]);
cuu   = reshape(c_ui_ui, [m m N]);
cuuj = c_ui_uj;
k     = zeros(m,N-1);
K     = zeros(m,n,N-1);
Ku     = zeros(size(D.Nodes,1),m,m,N-1);
Vx    = zeros(n,N);
Vxx   = zeros(n,n,N);
dV    = [0 0];

Vx(:,N)     = cx(:,N);
Vxx(:,:,N)  = cxx(:,:,N);

diverge  = 0;
for i = N-1:-1:1
    
    Qu  = cu(:,i)      + fu(:,:,i)'*Vx(:,i+1);              %(23)
    Qx  = cx(:,i)      + fx(:,:,i)'*Vx(:,i+1);              %(23)
    Qux = cxu(:,:,i)'  + fu(:,:,i)'*Vxx(:,:,i+1)*fx(:,:,i); %(24)
    if ~isempty(fxu)
        fxuVx = vectens(Vx(:,i+1),fxu(:,:,:,i));
        Qux   = Qux + fxuVx;
    end
    
    Quu = cuu(:,:,i)   + fu(:,:,i)'*Vxx(:,:,i+1)*fu(:,:,i); %(24)
    if ~isempty(fuu)
        fuuVx = vectens(Vx(:,i+1),fuu(:,:,:,i));
        Quu   = Quu + fuuVx;
    end
    
    Qxx = cxx(:,:,i)   + fx(:,:,i)'*Vxx(:,:,i+1)*fx(:,:,i); %(24)
    if ~isempty(fxx)
        Qxx = Qxx + vectens(Vx(:,i+1),fxx(:,:,:,i));
    end
    
    Vxx_reg = (Vxx(:,:,i+1) + lambda*eye(n)*(regType == 2));    %(48)
    
    Qux_reg = cxu(:,:,i)'   + fu(:,:,i)'*Vxx_reg*fx(:,:,i);
    if ~isempty(fxu)
        Qux_reg = Qux_reg + fxuVx;
    end
    % in belief regularization
    % regType == 1 doesnt follow (48), but is more simple
    QuuF = cuu(:,:,i)  + fu(:,:,i)'*Vxx_reg*fu(:,:,i) + lambda*eye(m)*(regType == 1);
    
    if ~isempty(fuu)
        QuuF = QuuF + fuuVx;
    end
    % usually we go to else
%     if nargin < 13 || isempty(lims) || lims(1,1) > lims(1,2)
%         % no control limits: Cholesky decomposition, check for non-PD
%         [R,d] = chol(QuuF);
%         if d ~= 0
%             diverge  = i;
%             return;
%         end
%         
%         % find control law
%         kK = -R\(R'\[Qu Qux_reg]);
%         k_i = kK(:,1);
%         K_i = kK(:,2:n+1);
%         
%     else        % solve Quadratic Program
%         lower = lims(:,1)-u(idx,:,i);
%         upper = lims(:,2)-u(:,i);
%         % 0.5*x'*H*x + x'*g  s.t. lower<=x<=upper
%         [k_i,result,R,free] = boxQP(QuuF,Qu,lower,upper,k(:,min(i+1,N-1)));
%         % R*R'=Quu (actually QuuF)
%         % k_i=-Quu\Qu
%         %(38)(42)
%         if result < 1
%             diverge  = i;
%             return;
%         end
%         
%         K_i    = zeros(m,n);
    K2_i=-QuuF\(cxu(:,:,i)+fx(:,:,i)'*Vxx_reg*fu(:,:,i))';
%         Vb_gu_plus_cu = cu(:,i)      + fu(:,:,i)'*(Vx(:,i+1) + Vxx(:,i+1)*);
    k2_i = -QuuF\Qu;
    incoming_nbrs_idces = predecessors(D,idx)';
    for j = incoming_nbrs_idces
%         Ku(j,:,:,i) = -QuuF\squeeze(c_ui_uj(j,:,:,i));
    end
%         if any(free)
%             Lfree        = -R\(R'\Qux_reg(free,:)); %=-(R*R')^(-1)*Qux=-Quu^(-1)*Qux
%             %(38)(42)
%             K_i(free,:)   = Lfree;
%             
%         end
        
%     end
    
    % update cost-to-go approximation
    dV          = dV + [k2_i'*Qu  .5*k2_i'*Quu*k2_i];              %(43)
    Vx(:,i)     = Qx  + K2_i'*Quu*k2_i + K2_i'*Qu  + Qux'*k2_i;     %(44)
    Vxx(:,:,i)  = Qxx + K2_i'*Quu*K2_i + K2_i'*Qux + Qux'*K2_i;     %(45)
    Vxx(:,:,i)  = .5*(Vxx(:,:,i) + Vxx(:,:,i)'); % ensure symmetry
    
    % save controls/gains
%     k(:,i)      = k_i;
	k(:,i)      = k2_i;
%     K(:,:,i)    = K_i;
    K(:,:,i)    = K2_i;
%     Ku(:,:,i) = Ku_i_agent_j
end% end of for (horizon)



function  stop = graphics(figures,x,u,cost,L,Vx,Vxx,fx,fxx,fu,fuu,trace,init)
stop = 0;

if figures == 0
    return;
end

n  = size(x,1);
N  = size(x,2);
nL = size(L,2);
m  = size(u,1);

cost  = sum(cost,1);
T     = [trace.iter];
T     = T(~isnan(T));
mT    = max(T);

% === first figure
if figures ~= 0  && ( mod(mT,figures) == 0 || init == 2 )
    
    fig1 = findobj(0,'name','iLQG');
    if  isempty(fig1)
        fig1 = figure();
        set(fig1,'NumberTitle','off','Name','iLQG','KeyPressFcn',@Kpress,'user',0,'toolbar','none', 'WindowStyle', 'docked');
        fprintf('Type ESC in the graphics window to terminate early.\n')
    end
    
    if mT == 1
        set(fig1,'user',0);
    end
    
    set(0,'currentfigure',fig1);
    clf(fig1);
    
    ax1   = subplot(2,2,1);
    set(ax1,'XAxisL','top','YAxisL','right','xlim',[1 N],'xtick',[])
    line(1:N,cost,'linewidth',4,'color',.5*[1 1 1]);
    ax2 = axes('Position',get(ax1,'Position'));
%     plot((1:N),x','linewidth',2);
    set(ax2,'xlim',[1 N],'Ygrid','on','YMinorGrid','off','color','none');
    set(ax1,'Position',get(ax2,'Position'));
    double_title(ax1,ax2,'state','running cost')
    
    axL = subplot(2,2,3);
    CO = get(axL,'colororder');
    set(axL,'nextplot','replacechildren','colororder',CO(1:min(n,7),:))
    Lp = reshape(permute(L,[2 1 3]), [nL*m N-1])';
%     plot(axL,1:N-1,Lp,'linewidth',1,'color',0.7*[1 1 1]);
    ylim  = get(axL,'Ylim');
    ylim  = [-1 1]*max(abs(ylim));
    set(axL,'XAxisL','top','YAxisL','right','xlim',[1 N],'xtick',[],'ylim',ylim)
    axu = axes('Position',get(axL,'Position'));
%     plot(axu,(1:N-1),u(:,1:N-1)','linewidth',2);
    ylim  = get(axu,'Ylim');
    ylim  = [-1 1]*max(abs(ylim));
    set(axu,'xlim',[1 N],'Ygrid','on','YMinorGrid','off','color','none','ylim',ylim);
    set(axL,'Position',get(axu,'Position'));
    double_title(axu,axL,'controls','gains')
    xlabel 'timesteps'
    
    ax1      = subplot(2,2,2);
    set(ax1,'XAxisL','top','YAxisL','right','xlim',[1 mT+eps],'xtick',[])
    hV = line(T,[trace(T).cost],'linewidth',4,'color',.5*[1 1 1]);
    ax2 = axes('Position',get(ax1,'Position'));
    converge = [[trace(T).lambda]' [trace(T).alpha]' [trace(T).grad_norm]' [trace(T).improvement]'];
    hT = semilogy(T,max(0, converge),'.-','linewidth',2,'markersize',10);
    set(ax2,'xlim',[1 mT+eps],'Ygrid','on','YMinorGrid','off','color','none');
    set(ax1,'Position',get(ax2,'Position'));
    double_title(ax1,ax2,'convergence trace','total cost')
    
    subplot(2,2,4);
%     plot(T,[trace(T).reduc_ratio]','.-','linewidth',2);
    title 'actual/expected reduction ratio'
    set(gca,'xlim',[0 mT+1],'ylim',[0 2],'Ygrid','on');
    xlabel 'iterations'
    
    set(findobj(fig1,'-property','FontSize'),'FontSize',8)
    stop = get(fig1,'user');
end

if figures < 0  &&  (mod(abs(trace(mT).iter)-1,figures) == 0 || init == 2) && ~isempty(Vx)
    
    fig2 = findobj(0,'name','iLQG - derivatives');
    if  isempty(fig2)
        fig2 = figure();
        set(fig2,'NumberTitle','off','Name','iLQG - derivatives','KeyPressFcn',@Kpress,'user',0, 'WindowStyle', 'docked');
    end
    
    if length(T) == 1
        set(fig2,'user',0);
    end
    
    set(0,'currentfigure',fig2);
    clf(fig2);
    
    subplot(2,3,1);
%     plot(1:N,Vx','linewidth',2);
    set(gca,'xlim',[1 N]);
    title 'V_x'
    grid on;
    
    subplot(2,3,4);
    z = reshape(Vxx,nL^2,N)';
    zd = (1:nL+1:nL^2);
%     plot(1:N,z(:,setdiff(1:nL^2,zd)),'color',.5*[1 1 1]);
    hold on;
%     plot(1:N,z(:,zd),'linewidth',2);
    hold off
    grid on;
    set(gca,'xlim',[1 N]);
    title 'V_{xx}'
    xlabel 'timesteps'
    
    subplot(2,3,2);
    Nfx     = size(fx,3);
    z = reshape(fx,nL^2,Nfx)';
    zd = (1:n+1:n^2);
%     plot(1:Nfx,z(:,setdiff(1:n^2,zd)),'color',.5*[1 1 1]);
    hold on;
%     plot(1:Nfx,z,'linewidth',2);
    set(gca,'xlim',[1 Nfx+eps]);
    hold off
    grid on;
    title 'f_{x}'
    
    if numel(fxx) > 0
        fxx = fxx(:,:,:,1:N-1);
        subplot(2,3,5);
        z  = reshape(fxx,[numel(fxx)/(N-1) N-1])';
%         plot(1:N-1,z);
        title 'f_{xx}'
        grid on;
        set(gca,'xlim',[1 N-1+eps]);
    end
    
    subplot(2,3,3);
    Nfu     = size(fu,3);
    z = reshape(fu,nL*m,Nfu)';
%     plot(1:Nfu,z','linewidth',2);
    set(gca,'xlim',[1 Nfu]);
    title 'f_u'
    grid on;
    
    if numel(fuu) > 0
        subplot(2,3,6);
        fuu = fuu(:,:,:,1:N-1);
        z  = reshape(fuu,[numel(fuu)/(N-1) N-1])';
%         plot(1:N-1,z);
        title 'f_{uu}'
        grid on;
        set(gca,'xlim',[1 N-1+eps]);
    end
    
    set(findobj(fig2,'-property','FontSize'),'FontSize',8)
    stop = stop + get(fig2,'user');
end

if init == 1
    figure(fig1);
elseif init == 2
    strings  = {'V','\lambda','\alpha','\partial_uV','\Delta{V}'};
    legend([hV; hT],strings,'Location','Best');
end

drawnow;

function Kpress(src,evnt)
if strcmp(evnt.Key,'escape')
    set(src,'user',1)
end

function double_title(ax1, ax2, title1, title2)

t1 = title(ax1, title1);
set(t1,'units','normalized')
pos1 = get(t1,'position');
t2 = title(ax2, title2);
set(t2,'units','normalized')
pos2 = get(t2,'position');
[pos1(2),pos2(2)] = deal(min(pos1(2),pos2(2)));
pos1(1)  = 0.05;
set(t1,'pos',pos1,'HorizontalAlignment','left')
pos2(1)  = 1-0.05;
set(t2,'pos',pos2,'HorizontalAlignment','right')

% setOpts - a utility function for setting default parameters
% ===============
% defaults  - either a cell array or a structure of field/default-value pairs.
% options   - either a cell array or a structure of values which override the defaults.
% params    - structure containing the union of fields in both inputs.
function params = setOpts(defaults,options)

if nargin==1 || isempty(options)
    user_fields  = [];
else
    if isstruct(options)
        user_fields   = fieldnames(options);
    else
        user_fields = options(1:2:end);
        options     = struct(options{:});
    end
end

if isstruct(defaults)
    params   = defaults;
else
    params   = struct(defaults{:});
end

for k = 1:length(user_fields)
    params.(user_fields{k}) = options.(user_fields{k});
end