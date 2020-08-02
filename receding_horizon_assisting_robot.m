%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario with a
% point robot whose body is modeled as a disk
% and it can sense beacons in the world.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function receding_horizon_assisting_robot()

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
EQUAL_WEIGHT_BALANCING = 1;
EQUAL_WEIGHT_TO_BALL_FEEDBACK = 2;
EQUAL_WEIGHT_TO_REST_FEEDBACK = 3;
BALL_WISH_WITHOUT_HUMAN_INPUT = 4;
BALL_WISH_WITH_HUMAN_INPUT = 5;
BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT = 6;
REST_WISH_WITHOUT_HUMAN_INPUT = 7;
REST_WISH_WITH_HUMAN_INPUT = 8;
REST_WISH_WITH_OPPOSITE_HUMAN_INPUT = 9;
show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
switch show_mode
    case EQUAL_WEIGHT_BALANCING
        weight_1 = 0.5;
        weight_2 = 0.5;
    case EQUAL_WEIGHT_TO_BALL_FEEDBACK
        weight_1 = 0.5;
        weight_2 = 0.5;
    case EQUAL_WEIGHT_TO_REST_FEEDBACK
        weight_1 = 0.5;
        weight_2 = 0.5;        
    case BALL_WISH_WITHOUT_HUMAN_INPUT
        weight_1 = 0.95;
        weight_2 = 0.05;
    case BALL_WISH_WITH_HUMAN_INPUT
        weight_1 = 0.95;
        weight_2 = 0.05;
    case BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_1 = 0.95;
        weight_2 = 0.05;
    case REST_WISH_WITHOUT_HUMAN_INPUT
        weight_1 = 0.05;
        weight_2 = 0.95;
    case REST_WISH_WITH_HUMAN_INPUT
        weight_1 = 0.05;
        weight_2 = 0.95;
    case REST_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_1 = 0.05;
        weight_2 = 0.95;
end
%% tuned parameters
mu_1 = [8.5, 4.0, 5.0, 0.0]';
mu_2 = [3, 2.0, 5.0, 0.0]';
sig_1 = diag([0.01, 0.01, 0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01, 0.01, 0.01]);
% weight_1 = 0.9;
% weight_2 = 0.1;
dt = 0.05;
horizon = 2.0;
mpc_update_period = 2.0;
simulation_time = 2;

%% 

t0 = 0;
tspan = t0 : dt : horizon;
horizonSteps = length(tspan);
tspan_btw_updates = t0 : dt : mpc_update_period;
update_steps = length(tspan_btw_updates);
simulation_steps = simulation_time/mpc_update_period;

mm = HumanMind(dt); % motion model

om = HumanReactionModel(); % observation model

%% Setup start and goal/target state

b0=[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];

u_guess = zeros(6,horizonSteps);
% initial guess, less iterations needed if given well
u_guess(1,:)=-3.3;
u_guess(2,:)=-1.3;

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

for i_sim = 1:simulation_steps
    [b,u_opt,L_opt,~,~,optimCost,~,~,tt, nIter]= iLQG_GMM(DYNCST, b0, u_guess, Op);
    if i_sim < 2
        show_mode = EQUAL_WEIGHT_TO_BALL_FEEDBACK;
    else
        show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
    end
    time_past = (i_sim-1) * mpc_update_period;
    [didCollide, b0, x_true_final] = mpc_animateGMM(5,6,b0, b, u_opt, L_opt, update_steps,time_past, mm, om,Op.lims, show_mode);
    b0(1:2) = x_true_final(1:2);
end

end


