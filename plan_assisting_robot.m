%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario with a
% point robot whose body is modeled as a disk
% and it can sense beacons in the world.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plan_assisting_robot()

addpath(genpath('./'));
clear
% FigList = findall(groot, 'Type', 'figure');
% for iFig = 1:numel(FigList)
%     try
%         clf(FigList(iFig));
%     catch
%         % Nothing to do
%     end
% end
%% tuned parameters
mu_1 = [8.5, 4.0, 5.0, 0.0]';
mu_2 = [3, 2.0, 5.0, 0.0]';
sig_1 = diag([0.01, 0.01, 0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01, 0.01, 0.01]);
weight_1 = 0.9;
weight_2 = 0.1;
dt = 0.05;
tf = 2.0;
%% 

t0 = 0;
tspan = t0 : dt : tf;
horizonSteps = length(tspan);

mm = HumanMind(dt); % motion model

om = HumanReactionModel(); % observation model

%% Setup start and goal/target state

b0=[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];

u0 = zeros(6,horizonSteps);
% initial guess, less iterations needed if given well
u0(1,:)=-3.3;
u0(2,:)=-1.3;

full_DDP = false;

% this function is needed by iLQG
% DYNCST  = @(b,u,i) beliefDynCost(b,u,xf,nDT,full_DDP,mm,om,svc);
DYNCST  = @(b,u,i) beliefDynCost_assisting_robot(b,u,horizonSteps,full_DDP,mm,om);
% control constraints are optional
Op.lims  = [-0.0 0.0;
    -4.0 4.0;
    -0.0  0.0;
    -0.0  0.0;
    -2.0 2.0;
    -2.0 2.0];

Op.plot = -1; % plot the derivatives as well

%% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','r','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

%% === run the optimization
[b,u_opt,L_opt,~,~,optimCost,~,~,tt, nIter]= iLQG_GMM(DYNCST, b0, u0, Op);
assignin('base', 'b0', b0)
assignin('base', 'b', b)
assignin('base', 'u_opt', u_opt)
assignin('base', 'L_opt', L_opt)
assignin('base', 'mm', mm)
assignin('base', 'om', om)
assignin('base', 'lims', Op.lims)
lims = Op.lims;
[didCollide, b_f] = animateGMM(5,6,b0, b, u_opt, L_opt, size(b,2), mm, om,lims);

results.collision{1} = didCollide;


end


