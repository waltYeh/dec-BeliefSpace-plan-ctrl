%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2015, Yuval Tassa
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Washington nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, u, L, Vx, Vxx, cost, trace, stop, totaTime, ...
    totalIterations] = iLQG_multiagent(D,idx,DYNCST, x0, u0, Op)

% iLQG - solve the deterministic finite-horizon optimal control problem.
%
%        minimize sum_i CST(x(:,i),u(:,i)) + CST(x(:,end))
%            u
%        s.t.  x(:,i+1) = DYN(x(:,i),u(:,i))
%
% Inputs
% ======
% DYNCST - A combined dynamics and cost function. It is called in
% three different formats.
%
%  1) step:
%   [xnew,c] = DYNCST(x,u,i) is called during the forward pass. 
%   Here the state x and control u are vectors: size(x)==[n 1],  
%   size(u)==[m 1]. The cost c and time index i are scalars.
%   If Op.parallel==true (the default) then DYNCST(x,u,i) is be 
%   assumed to accept vectorized inputs: size(x,2)==size(u,2)==K
%  
%  2) final:
%   [~,cnew] = DYNCST(x,nan) is called at the end the forward pass to compute
%   the final cost. The nans indicate that no controls are applied.
%  
%  3) derivatives:
%   [~,~,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = DYNCST(x,u,I) computes the
%   derivatives along a trajectory. In this case size(x)==[n N+1] where N
%   is the trajectory length. size(u)==[m N+1] with NaNs in the last column
%   to indicate final-cost. The time indexes are I=(1:N).
%   Dimensions match the variable names e.g. size(fxu)==[n n m N+1]
%   note that the last temporal element N+1 is ignored for all tensors
%   except cx and cxx, the final-cost derivatives.
%
% x0 - The initial state from which to solve the control problem. 
% Should be a column vector. If a pre-rolled trajectory is available
% then size(x0)==[n N+1] can be provided and Op.cost set accordingly.
%
% u0 - The initial control sequence. A matrix of size(u0)==[m N]
% where m is the dimension of the control and N is the number of state
% transitions. 
%
%
% Op - optional parameters, see below
%
% Outputs
% =======
% x - the optimal state trajectory found by the algorithm.
%     size(x)==[n N+1]
%
% u - the optimal open-loop control sequence.
%     size(u)==[m N]
%
% L - the optimal closed loop control gains. These gains multiply the
%     deviation of a simulated trajectory from the nominal trajectory x.
%     size(L)==[m n N]
%
% Vx - the gradient of the cost-to-go. size(Vx)==[n N+1]
%
% Vxx - the Hessian of the cost-to-go. size(Vxx)==[n n N+1]
%
% cost - the costs along the trajectory. size(cost)==[1 N+1]
%        the cost-to-go is V = fliplr(cumsum(fliplr(cost)))
%
% lambda - the final value of the regularization parameter
%
% trace - a trace of various convergence-related values. One row for each
%         iteration, the columns of trace are
%         [iter lambda alpha g_norm dcost z sum(cost) dlambda]
%         see below for details.
%
% timing - timing information

%---------------------- user-adjustable parameters ------------------------
defaults = {'lims',           [],...            control limits
            'parallel',       true,...          use parallel line-search?
            'Alpha',          10.^linspace(0,-3,11),... backtracking coefficients
            'tolFun',         1e-1,...          reduction exit criterion
            'tolGrad',        1e-4,...          gradient exit criterion
            'maxIter',        50,...           maximum iterations            
            'lambda',         1,...             initial value for lambda
            'dlambda',        1,...             initial value for dlambda
            'lambdaFactor',   1.4,...           lambda scaling factor
            'lambdaMax',      1e16,...          lambda maximum value
            'lambdaMin',      1e-6,...          below this value lambda = 0
            'regType',        1,...             regularization type 1: q_uu+lambda*eye(); 2: V_xx+lambda*eye()
            'zMin',           0,...             minimal accepted reduction ratio
            'diffFn',         [],...            user-defined diff for sub-space optimization
            'plot',           0,...             0: no;  k>0: every k iters; k<0: every k iters, with derivs window
            'print',          3,...             0: no;  1: final; 2: iter; 3: iter, detailed
            'plotFn',         @(x)0,...         user-defined graphics callback
            'cost',           [],...            initial cost for pre-rolled trajectory            
            };

% --- initial sizes and controls
n   = size(x0, 2);          % dimension of belief state vector
m   = size(u0, 2);          % dimension of control vector
N   = size(u0, 3);          % number of state transitions time steps
u   = u0;                   % initial control sequence

% --- proccess options
if nargin < 4
    Op = struct();
end
Op  = setOpts(defaults,Op);

verbosity = Op.print;

switch numel(Op.lims)
    case 0
    case 2*m
        % we are here, no change in fact
        Op.lims = sort(Op.lims,2);
    case 2
        Op.lims = ones(m,1)*sort(Op.lims(:))';
    case m
        Op.lims = Op.lims(:)*[-1 1];
    otherwise
        error('limits are of the wrong size')
end

lambda   = Op.lambda;
dlambda  = Op.dlambda;

% --- initialize trace data structure
trace = struct('iter',nan,'lambda',nan,'dlambda',nan,'cost',nan,...
        'alpha',nan,'grad_norm',nan,'improvement',nan,'reduc_ratio',nan,...
        'time_derivs',nan,'time_forward',nan,'time_backward',nan);
trace = repmat(trace,[min(Op.maxIter,1e6) 1]);
trace(1).iter = 1;
trace(1).lambda = lambda;
trace(1).dlambda = dlambda;

% --- initial belief state given
% if size(x0,3) == 1
    diverge = true;
    for alpha = Op.Alpha
        [x,un,cost]  = forward_pass(D,idx,x0,alpha*u,[],[],[],[],1,DYNCST,Op.lims,[]);
%         drawResult(Op.plotFn,x(:,:,1),2);
%         saveas(gcf,'iLQG-initialguess.jpg');
%         pause(3);
        % simplistic divergence test
        if all(abs(x(:)) < 1e8)
            u = un;
            diverge = false;
            break
        end
    end
% elseif size(x0,3) == N+1 % pre-rolled initial forward pass
%     x        = x0;
%     diverge  = false;
%     if isempty(Op.cost)
%         error('pre-rolled initial trajectory requires cost')
%     else
%         cost     = Op.cost;
%     end
% else
%     error('pre-rolled initial trajectory must be of correct length')
% end

trace(1).cost = sum(cost(:));

% user plotting
% drawResult(Op.plotFn,x,2);
% Op.plotFn(x);

if diverge
    [Vx,Vxx, stop]  = deal(nan);
    L        = zeros(m,n,N);
    cost     = [];
    trace    = trace(1);
    if verbosity > 0
        fprintf('\nEXIT: Initial control sequence caused divergence\n');
    end
    return
end

% constants, timers, counters
flgChange   = 1;
stop        = 0;
dcost       = 0;
z           = 0;
expected    = 0;
print_head  = 6; % print headings every print_head lines
last_head   = print_head;
t_start     = tic;
if verbosity > 0
    fprintf('\n=========== begin iLQG ===========\n');
end
% graphics(Op.plot,x,u,cost,zeros(m,n,N),[],[],[],[],[],[],trace,1);
for iter = 1:Op.maxIter
    if stop
        break;
    end
    trace(iter).iter = iter;    
    
    %====== STEP 1: differentiate dynamics and cost along new trajectory
    if flgChange
        t_diff = tic;
        [~,~,fx,fu,fxx,fxu,fuu,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj]   = DYNCST(D,idx,x, cat(3,u,nan(size(D.Nodes,1),m,1)), 1:N+1);
        trace(iter).time_derivs = toc(t_diff);
        flgChange   = 0;
    end
    
    %====== STEP 2: backward pass, compute optimal control law and cost-to-go
    backPassDone   = 0;
    while ~backPassDone
        
        t_back   = tic;
        [diverge, Vx, Vxx, l, L, dV, Ku] = back_pass(D,idx,c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,fx,fu,fxx,fxu,fuu,lambda,Op.regType,Op.lims,u);
        % l is the feedforward term (42), 
        % L is the time-variant feedback(42)
        trace(iter).time_backward = toc(t_back);
        
        if diverge
            % maybe not step 13 in Algorithm
            if verbosity > 2
                fprintf('Cholesky failed at timestep %d.\n',diverge);
            end
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
    % another exit criteria than Schwarting
    g_norm         = mean(max(abs(l) ./ (squeeze(abs(u(idx,:,:)+1))),[],1));
    trace(iter).grad_norm = g_norm;
    if g_norm < Op.tolGrad && lambda < 1e-5
        dlambda   = min(dlambda / Op.lambdaFactor, 1/Op.lambdaFactor);
        lambda    = lambda * dlambda * (lambda > Op.lambdaMin);
        if verbosity > 0
            fprintf('\nSUCCESS: gradient norm < tolGrad\n');
        end
        break;
    end
    
    %====== STEP 3: line-search to find new control sequence, trajectory, cost
    fwdPassDone  = 0;
    if backPassDone
        t_fwd = tic;
        if Op.parallel  % parallel line-search
            [xnew,unew,costnew] = forward_pass(D,idx,x0 ,u, L, x(:,:,1:N), l, Ku, Op.Alpha, DYNCST,Op.lims,Op.diffFn);
            % now we have 10 candidates of new traj
            Dcost               = sum(cost(:)) - sum(squeeze(costnew),1);
            [dcost, w]          = max(Dcost);
            % find the one with greatest cost reduction
            alpha               = Op.Alpha(w);
            expected            = -alpha*(dV(1) + alpha*dV(2));
            if expected > 0
                z = dcost/expected;
            else
                z = sign(dcost);
                warning('non-positive expected reduction: should not occur');
            end
            if (z > Op.zMin)
                fwdPassDone = 1;
                costnew     = costnew(:,:,:,w);
                xnew        = xnew(:,:,:,w);
                unew        = unew(:,:,:,w);
            end
        else            % serial backtracking line-search
            for alpha = Op.Alpha
                [xnew,unew,costnew]   = forward_pass(D,idx,x0 ,u+l*alpha, L, x(:,:,1:N),[],Ku,1,DYNCST,Op.lims,Op.diffFn);
                dcost    = sum(cost(:)) - sum(costnew(:));
                expected = -alpha*(dV(1) + alpha*dV(2));
                if expected > 0
                    z = dcost/expected;
                else
                    z = sign(dcost);
                    warning('non-positive expected reduction: should not occur');
                end
                if (z > Op.zMin)
                    fwdPassDone = 1;
                    break;
                end
            end
        end
        if ~fwdPassDone
            alpha = nan; % signals failure of forward pass
        end
        trace(iter).time_forward = toc(t_fwd);
    end
    
    %====== STEP 4: accept step (or not), draw graphics, print status
    
    % print headings
    if verbosity > 1 && last_head == print_head
        last_head = 0;
        fprintf('%-12s','iteration','cost','reduction','expected','gradient','log10(lambda)')
        fprintf('\n');
    end
    
    if fwdPassDone
        
        % print status
        if verbosity > 1
            fprintf('%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f\n', ...
                iter, sum(cost(:)), dcost, expected, g_norm, log10(lambda));
            last_head = last_head+1;
        end
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
        if dcost < Op.tolFun
            if verbosity > 0
                fprintf('\nSUCCESS: cost change < tolFun\n');
            end
            break;
        end
        
    else % no cost improvement
        % increase lambda
        dlambda  = max(dlambda * Op.lambdaFactor, Op.lambdaFactor);
        lambda   = max(lambda * dlambda, Op.lambdaMin);
        
        % print status
        if verbosity > 1
            fprintf('%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f\n', ...
                iter,'NO STEP', dcost, expected, g_norm, log10(lambda));           
            last_head = last_head+1;
        end     
        
        % terminate ?
        if lambda > Op.lambdaMax,
            if verbosity > 0
                fprintf('\nEXIT: lambda > lambdaMax\n');
            end
            break;
        end
    end
    % update trace
    trace(iter).lambda      = lambda;
    trace(iter).dlambda     = dlambda;
    trace(iter).alpha       = alpha;
    trace(iter).improvement = dcost;
    trace(iter).cost        = sum(cost(:));
    trace(iter).reduc_ratio = z;
%     stop = graphics(Op.plot,x,u,cost,L,Vx,Vxx,fx,fxx,fu,fuu,trace(1:iter),0);
end

% save lambda/dlambda
trace(iter).lambda      = lambda;
trace(iter).dlambda     = dlambda;

% if stop
%     if verbosity > 0
%         fprintf('\nEXIT: Terminated by user\n');
%     end
% end

if iter == Op.maxIter
    if verbosity > 0
        fprintf('\nEXIT: Maximum iterations reached.\n');
    end
end


if ~isempty(iter)
    diff_t = [trace(1:iter).time_derivs];
    diff_t = sum(diff_t(~isnan(diff_t)));
    back_t = [trace(1:iter).time_backward];
    back_t = sum(back_t(~isnan(back_t)));
    fwd_t = [trace(1:iter).time_forward];
    fwd_t = sum(fwd_t(~isnan(fwd_t)));
    total_t = toc(t_start);
    totalIterations = iter;
    if verbosity > 0
        fprintf(['\n'...
            'iterations:   %-3d\n'...
            'final cost:   %-12.7g\n' ...
            'final grad:   %-12.7g\n' ...
            'final lambda: %-12.7e\n' ...
            'time / iter:  %-5.0f ms\n'...
            'total time:   %-5.2f seconds, of which\n'...
            '  derivs:     %-4.1f%%\n'...
            '  back pass:  %-4.1f%%\n'...
            '  fwd pass:   %-4.1f%%\n'...
            '  other:      %-4.1f%% (graphics etc.)\n'...
            '=========== end iLQG ===========\n'],...
            iter,sum(cost(:)),g_norm,lambda,1e3*total_t/iter,total_t,...
            [diff_t, back_t, fwd_t, (total_t-diff_t-back_t-fwd_t)]*100/total_t);
    end
    totaTime = total_t;
    trace    = trace(~isnan([trace.iter]));
%     timing   = [diff_t back_t fwd_t total_t-diff_t-back_t-fwd_t];
%     graphics(Op.plot,x,u,cost,L,Vx,Vxx,fx,fxx,fu,fuu,trace,2); % draw legend
else
    error('Failure: no iterations completed, something is wrong.')
end

% [x,un,cost]  = forward_pass(x0(:,1),alpha*u,[],[],[],1,DYNCST,Op.lims,[]);
function [xnew,unew,cnew] = forward_pass(D,idx,x0,u,L,x,du,Ku,Alpha,DYNCST,lims,diff)
% l (Schwarting j_k^i) is taken into the function as the argument du
% parallel forward-pass (rollout)
% internally time is on the 3rd dimension, 
% to facillitate vectorized dynamics calls
% Alpha is a series of possible step length, 
% it is multiplied with du(:,i) in parallel
n_agent = size(x0,1);
n_b        = size(x0,2);
K        = length(Alpha);
K1       = ones(1,K); % useful for expansion
m_u        = size(u,2);
N        = size(u,3);

xnew        = zeros(n_agent,n_b,K,N);
xnew(:,:,:,1) = x0(:,:,ones(1,K));
unew        = zeros(n_agent,m_u,K,N);
cnew        = zeros(1,1,K,N+1);% one agent, one dim c
for k = 1:N
    unew(:,:,:,k) = u(:,:,k*K1);
    
    if ~isempty(du)
        % feedforward control term should not be too agressive, Alpha < 1
        unew(idx,:,:,k) = squeeze(unew(idx,:,:,k)) + du(:,k)*Alpha;
    end    
    
    if ~isempty(L)
        if ~isempty(diff)
            dx = diff(xnew(:,:,k), x(:,k*K1));
        else
            dx          = xnew(idx,:,:,k) - x(idx,:,k*K1);
        end
        unew(idx,:,:,k) = squeeze(unew(idx,:,:,k)) + L(:,:,k)*squeeze(dx); % with feedback
    end
    if ~isempty(Ku)
        du_all_agent = u;
%         u_real_all_agent = u;
        du_all_agent(:) = 0;
%         dx          = xnew(idx,:,:,k) - x(idx,:,k*K1);
        incoming_nbrs = predecessors(D,idx);
        for j_agent=incoming_nbrs
            unew(idx,:,:,k) = squeeze(unew(idx,:,:,k)) + squeeze(Ku(j_agent,:,:,k))'*transpose(squeeze(du_all_agent(j_agent,:,k)));
        end
         % with feedback
    end
    if ~isempty(lims)
        % add u upper and lower bound again
        for i=1:size(D.Nodes,1)
            unew_ik = squeeze(unew(i,:,:,k));
            size_unew = size(unew_ik);
            if size_unew(2)==m_u
                unew_ik = transpose(unew_ik);
            end
            unew(i,:,:,k) = min(lims(:,2*K1), max(lims(:,1*K1), unew_ik));
        end
    end

    [xnew(:,:,:,k+1), cnew(:,:,:,k)]  = DYNCST(D,idx,xnew(:,:,:,k), unew(:,:,:,k), k*K1);
%     
end
[~, cnew(:,:,:,N+1)] = DYNCST(D,idx,xnew(:,:,:,N+1),nan(n_agent,m_u,K,1),k);
xnew = permute(xnew, [1 2 4 3]);
unew = permute(unew, [1 2 4 3]);
cnew = permute(cnew, [1 2 4 3]);
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
    incoming_nbrs_idces = predecessors(D,idx);
    for j = incoming_nbrs_idces
        Ku(j,:,:,i) = -QuuF\squeeze(c_ui_uj(j,:,:,i));
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
