%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario with a
% point robot whose body is modeled as a disk
% and it can sense beacons in the world.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plan_assisting_robot()

addpath(genpath('./'));
clear
close all
t0 = 0;
dt = 0.05;
tf = 1.0;
tspan = t0 : dt : tf;
horizonSteps = length(tspan);

mm = HumanMind(dt); % motion model

om = HumanReactionModel(); % observation model

%% Setup start and goal/target state
mu_1 = [8.5, 4.0, 5.0, 0.0]';
mu_2 = [3, 2.0, 5.0, 0.0]';
sig_1 = diag([0.01, 0.01, 0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01, 0.01, 0.01]);
weight_1 = 0.6;
weight_2 = 0.4;
b0=[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];

% [failed, b_f] = animateGMM(b0, nSteps, motionModel, obsModel);
%% Initialize planning scenario
DYNAMIC_OBS = 0;

% 
% global ROBOT_RADIUS;
% ROBOT_RADIUS = 0.46; % robot radius is needed by collision checker
% 
% svc = @(x)isStateValid(x,map,0); % state validity checker (collision)
% .goal; % target state

%% Setup planner to get nominal controls
% planner = RRT(map,mm,svc);
% % planner = StraightLine(map,mm,svc);
% 
% [~,u0, initGuessFigure] = planner.plan(x0,xf);
% 
% nDT = size(u0,2); % Time steps
u0 = zeros(6,horizonSteps);
u0(1,:)=-3.3;
u0(2,:)=-1.3;
%% set up the optimization problem

% Set full_DDP=true to compute 2nd order derivatives of the
% dynamics. This will make iterations more expensive, but
% final convergence will be much faster (quadratic)
full_DDP = false;

% this function is needed by iLQG
% DYNCST  = @(b,u,i) beliefDynCost(b,u,xf,nDT,full_DDP,mm,om,svc);
DYNCST  = @(b,u,i) beliefDynCost_assisting_robot(b,u,horizonSteps,full_DDP,mm,om);
% control constraints are optional
Op.lims  = [-0.1 0.1;
    -4.0 4.0;
    -1.0  1.0;
    -1.0  1.0;
    -2.0 2.0;
    -2.0 2.0];

Op.plot = -1; % plot the derivatives as well

%% prepare the visualization window and graphics callback
% figh = figure;
% set(figh,'WindowStyle','docked');
% drawLandmarks(figh,map.landmarks);
% drawObstacles(figh,map);
% scatter(x0(1),x0(2),250,'filled','MarkerFaceAlpha',1/2,'MarkerFaceColor',[1.0 0.0 0.0])
% scatter(xf(1),xf(2),250,'filled','MarkerFaceAlpha',1/2,'MarkerFaceColor',[0.0 1.0 0.0])
% set(gcf,'name','Belief Space Planning with iLQG','NumberT','off');
% set(gca,'Color',[0.0 0.0 0.0]);
% set(gca,'xlim',map.bounds(1,[1,2]),'ylim',map.bounds(2,[1,3]),'DataAspectRatio',[1 1 1])
% xlabel('X (m)'); ylabel('Y (m)');
% box on

%% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','r','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

%% === run the optimization
[b,u_opt,L_opt,~,~,optimCost,~,~,tt, nIter]= iLQG_GMM(DYNCST, b0, u0, Op);

[didCollide, b_f] = animateGMM(b0, b, u_opt, L_opt, size(b,2), mm, om);

results.collision{1} = didCollide;


end


