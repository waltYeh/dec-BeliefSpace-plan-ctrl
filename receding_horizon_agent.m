%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function receding_horizon_agent()
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
show_mode = EQUAL_WEIGHT_TO_BALL_FEEDBACK;
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
mu_1 = [0, 4.0]';
mu_2 = [1, 3.0]';
mu_3 = [-1, 2.8]';
mu_4 = [0.1, 2.0]';
sig_1 = diag([0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01]);
sig_3 = diag([0.01, 0.01]);%sigma
sig_4 = diag([0.01, 0.01]);
% weight_1 = 0.9;
% weight_2 = 0.1;
dt = 0.05;
horizon = 2.0;
mpc_update_period = 0.5;
simulation_time = 2;

%% 

t0 = 0;
tspan = t0 : dt : horizon;
horizonSteps = length(tspan);
tspan_btw_updates = t0 : dt : mpc_update_period;
update_steps = length(tspan_btw_updates);
simulation_steps = simulation_time/mpc_update_period;

% mm = HumanMind(dt); % motion model

% om = HumanReactionModel(); % observation model



sd = [2 2 3 1];%edges start from
td = [1 3 4 3];%edges go to
nom_formation_1=[0.4,0.4;
    -0.4,-0.4;
    -0.4,-0.4;
    -0.8,-0.8];%-- formation
nom_formation_2=[-0.5,0.6;
    -1,0;
    0.5,-0.6;
    -0.5,-0.6];%z formation
q_formation=[1;1;1;1];
rij_control = [0.3;0.3;0.3;0.3];%control cost of node sd in opt of td
rii_control = [0.8;0.8;0.8;0.8];
EdgeTable = table([sd' td'],nom_formation_1,nom_formation_2,q_formation,rij_control,'VariableNames',{'EndNodes' 'nom_formation_1' 'nom_formation_2' 'q_formation' 'rij_control'});
NodeTable = table(rii_control,'VariableNames',{'rii_control'});
D = digraph(EdgeTable,NodeTable);

agents = cell(size(D.Nodes,1),1);
agents{1} = AgentCrane(dt,horizonSteps,1);
agents{2} = AgentPlattform(dt,horizonSteps);
agents{3} = AgentCrane(dt,horizonSteps,3);
agents{4} = AgentCrane(dt,horizonSteps,4);

%% Setup start and goal/target state

u_guess = zeros(size(D.Nodes,1),size(D.Nodes,1),2,horizonSteps-1);
% initial guess, less iterations needed if given well
% u_guess(1,:)=-3.3;
% u_guess(2,:)=-1.3;
b0=zeros(size(D.Nodes,1),size(D.Nodes,1),size(mu_1,1)+size(sig_1(:),1));
for i=1:size(D.Nodes,1)
    b0(i,1,:) = [mu_1;sig_1(:)];
    b0(i,2,:) = [mu_2;sig_2(:)];
    b0(i,3,:) = [mu_3;sig_3(:)];
    b0(i,4,:) = [mu_4;sig_4(:)];
end
% b0={[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2]};

% control constraints are optional
Op.lims  = [-4.0 4.0;
    -4.0 4.0];
%% these are old codes remained
Op.plot = -1; % plot the derivatives as well

% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','r','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

%% === run the optimization

for i_sim = 1:simulation_steps
    [b_nom1,u_nom1,L_opt1,Vx1,Vxx1,cost1] = agents{1}.iLQG_agent(D,squeeze(b0(1,:,:)), squeeze(u_guess(1,:,:,:)), Op);
    agents{1}.updatePolicy(b_nom1,u_nom1,L_opt1);
%     [b_nom2,u_nom2,L_opt2,Vx2,Vxx2,cost2] = agents{2}.iLQG_GMM(b0{2}, u_guess, Op);
%     agents{2}.updatePolicy(b_nom2,u_nom2,L_opt2);
    if i_sim < 2
        show_mode = EQUAL_WEIGHT_TO_BALL_FEEDBACK;
    else
        show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
    end
    time_past = (i_sim-1) * mpc_update_period;
    
    [~, b0, x_true_final] = animateMultiagent(agents, b0, update_steps,time_past, show_mode);
    b0{1}(1:2) = x_true_final(1:2);
end

end
