%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function run_centralized()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario with a
% point robot whose body is modeled as a disk
% and it can sense beacons in the world.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all

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
        weight_1 = 0.499;
        weight_2 = 0.501;
    case EQUAL_WEIGHT_TO_BALL_FEEDBACK
        weight_1 = 0.5;
        weight_2 = 0.5;
    case EQUAL_WEIGHT_TO_REST_FEEDBACK
        weight_1 = 0.5;
        weight_2 = 0.5;        
    case BALL_WISH_WITHOUT_HUMAN_INPUT
        weight_1 = 0.99;
        weight_2 = 0.01;
    case BALL_WISH_WITH_HUMAN_INPUT
        weight_1 = 0.95;
        weight_2 = 0.05;
    case BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_1 = 0.95;
        weight_2 = 0.05;
    case REST_WISH_WITHOUT_HUMAN_INPUT
        weight_1 = 0.01;
        weight_2 = 0.99;
    case REST_WISH_WITH_HUMAN_INPUT
        weight_1 = 0.01;
        weight_2 = 0.99;
    case REST_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_1 = 0.01;
        weight_2 = 0.99;
end
%% tuned parameters
knowledge_gen=true;
% mu_1 = [8.5, 0.0, 5.0, 0.0]';
% mu_2 = [3, 1.0, 5.0, 0.0]';
% sig_1 = diag([0.01, 0.01, 0.1, 0.1]);%sigma
% sig_2 = diag([0.01, 0.01, 0.1, 0.1]);
if knowledge_gen
    mu_a1 = [8.5, 0.0, 5.0, 0.0]';
else
    mu_a1 = [8.5, 4.0, 5.0, 0.0]';
end
mu_a2 = [3, 1, 5.0, 0.0]';
mu_e = [6.0, 4]';
% if show_mode>=4 &&show_mode<=6
%     
%     mu_b = [4.3, -1.55]';
%     mu_c = [3.6, 1.25]';
%     mu_d = [6.1, 1.3]';
%     
% elseif show_mode>=7 &&show_mode<=9
    mu_b = [5, -1]';
    mu_c = [4, 1]';
    mu_d = [6, 1]';
% end
sig_a1 = diag([0.01, 0.01, 0.05, 0.05]);%sigma
sig_a2 = diag([0.01, 0.01, 0.05, 0.05]);
sig_b = diag([0.02, 0.02]);%sigma
sig_c = diag([0.02, 0.02]);
sig_d = diag([0.02, 0.02]);
sig_e = diag([0.02, 0.02]);
% weight_1 = 0.9;
% weight_2 = 0.1;
dt = 0.05;
horizon = 2.5;
mpc_update_period = 2.5;
simulation_time = 2.5;

%% 

t0 = 0;
tspan = t0 : dt : horizon;
horizonSteps = length(tspan);
tspan_btw_updates = t0 : dt : mpc_update_period;
update_steps = length(tspan_btw_updates);
simulation_steps = simulation_time/mpc_update_period;

mm = HumanMind(dt); % motion model
if knowledge_gen
    om = HumanReactionModel(); % observation model
else
    om = HumanReactionModel_homo();
end
%% Setup start and goal/target state

b0=[mu_a1;sig_a1(:);weight_1;mu_a2;sig_a2(:);weight_2;mu_b;sig_b(:);mu_c;sig_c(:);mu_d;sig_d(:);mu_e;sig_e(:)];

u_guess = zeros(12,horizonSteps-1);
% initial guess, less iterations needed if given well
u_guess(1,:)=0;
u_guess(2,:)=0;

full_DDP = false;

% this function is needed by iLQG
% DYNCST  = @(b,u,i) beliefDynCost(b,u,xf,nDT,full_DDP,mm,om,svc);
DYNCST  = @(b,u,i) beliefDynCost_assisting_robot_centralized(b,u,horizonSteps,full_DDP,mm,om);
% control constraints are optional
Op.lims  = [-0.0 0.0;%target A x
    -2.0 2.0;%target A y
    -0.0  0.0;%target B x
    -0.0  0.0;%target B y
    -4.0 4.0;%assist 2 x
    -4.0 4.0;%assist 2 y
    -4.0 4.0;%assist 3 x
    -4.0 4.0;%assist 3 y
    -4.0 4.0;%assist 4 x
    -4.0 4.0;%assist 4 y
    -4.0 4.0;%compl 5 x
    -4.0 4.0;%compl 5 y
    ];

Op.plot = -1; % plot the derivatives as well

%% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','r','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

%% === run the optimization

for i_sim = 1:simulation_steps
    [b_nom,u_nom,L_opt,~,~,optimCost,~,~,tt, nIter]= iLQG_GMM(DYNCST, b0, u_guess, Op);
%     if i_sim < 2
%         show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
%     else
%         show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
%     end
    time_past = (i_sim-1) * mpc_update_period;
%     assignin('base', 'interfDiGr', interfDiGr)
    assignin('base', 'b0_centr', b0)
    assignin('base', 'b_nom', b_nom)
    assignin('base', 'u_nom', u_nom)
    assignin('base', 'L_opt', L_opt)
    assignin('base', 'update_steps', update_steps)
    assignin('base', 'time_past', time_past)
    assignin('base', 'show_mode', show_mode)
    assignin('base', 'dt', dt)
    assignin('base', 'Op', Op)
    assignin('base', 'mm', mm)
    assignin('base', 'om', om)
%     assignin('base', 'x_true', x_true)
    [didCollide, b0_next, x_true_final] = mpc_centralized_animateGMM(31,32,b0_centr, b_nom, ...
        u_nom, L_opt, update_steps,time_past, mm, om,Op.lims, show_mode);
    b0_next(1:2) = x_true_final(1:2);
end

end


