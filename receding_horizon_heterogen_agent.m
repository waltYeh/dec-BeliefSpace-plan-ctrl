%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for a 2D belief space planning scenario 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function receding_horizon_heterogen_agent()
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
        weight_a1 = 0.5;
        weight_a2 = 0.5;
    case EQUAL_WEIGHT_TO_BALL_FEEDBACK
        weight_a1 = 0.5;
        weight_a2 = 0.5;
    case EQUAL_WEIGHT_TO_REST_FEEDBACK
        weight_a1 = 0.5;
        weight_a2 = 0.5;        
    case BALL_WISH_WITHOUT_HUMAN_INPUT
        weight_a1 = 0.95;
        weight_a2 = 0.05;
    case BALL_WISH_WITH_HUMAN_INPUT
        weight_a1 = 0.95;
        weight_a2 = 0.05;
    case BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_a1 = 0.95;
        weight_a2 = 0.05;
    case REST_WISH_WITHOUT_HUMAN_INPUT
        weight_a1 = 0.05;
        weight_a2 = 0.95;
    case REST_WISH_WITH_HUMAN_INPUT
        weight_a1 = 0.05;
        weight_a2 = 0.95;
    case REST_WISH_WITH_OPPOSITE_HUMAN_INPUT
        weight_a1 = 0.05;
        weight_a2 = 0.95;
end
%% tuned parameters
mu_a1 = [8.5, 0.0, 5.0, 0.0]';
mu_a2 = [3, 1.0, 5.0, 0.0]';
mu_b = [8.5, 0.0]';
mu_c = [5, 3]';
mu_d = [0.0, 0.0]';
sig_a1 = diag([0.01, 0.01, 0.1, 0.1]);%sigma
sig_a2 = diag([0.01, 0.01, 0.1, 0.1]);
sig_b = diag([0.01, 0.01]);%sigma
sig_c = diag([0.5, 0.5]);
sig_d = diag([0.5, 0.5]);

% weight_1 = 0.9;
% weight_2 = 0.1;
dt = 0.05;
horizon = 3.0;
mpc_update_period = 1;
simulation_time = 3;

%% 

t0 = 0;
tspan = t0 : dt : horizon;
horizonSteps = length(tspan);
tspan_btw_updates = t0 : dt : mpc_update_period;
update_steps = length(tspan_btw_updates);
simulation_steps = simulation_time/mpc_update_period;

% mm = HumanMind(dt); % motion model

% om = HumanReactionModel(); % observation model



sd = [2 2 3 1 1];%edges start from
td = [1 3 4 3 2];%edges go to
nom_formation_2=[0.5,0.5;
    -0.5,-0.5;
    -0.5,-0.5;
    -1,-1;
    -0.5,-0.5;];%-- formation
nom_formation_2=[-1,1;
    -2,0;
    1,-1;
    -1,-1;
    1,-1];%z formation
q_formation=[1;1;1;1;1];
rij_control = [0.3;0.3;0.3;0.3;0.3];%control cost of node sd in opt of td
rii_control = [0.8;0.8;0.8;0.8];
incoming_edges = zeros(4,4);
EdgeTable = table([sd' td'],nom_formation_2,q_formation,rij_control,'VariableNames',{'EndNodes' 'nom_formation_2' 'q_formation' 'rij_control'});

NodeTable = table(incoming_edges,rii_control,'VariableNames',{'incoming_edges' 'rii_control'});
interfDiGr = digraph(EdgeTable,NodeTable);
for idx=1:4
    incoming_nbrs_idces = predecessors(interfDiGr,idx)';
    for j = incoming_nbrs_idces
        if isempty(j)
            continue
        end
        RowIdx = ismember(interfDiGr.Edges.EndNodes, [j,idx],'rows');
        [rId, cId] = find( RowIdx ) ;
        interfDiGr.Nodes.incoming_edges(idx,j)=rId;
    end
end

comm_sd = [1 1 1 3];
comm_td = [2 3 4 4];
commGr = graph(comm_sd,comm_td);
adjGr = full(adjacency(commGr));% full transfers sparse to normal matrix

agents = cell(size(interfDiGr.Nodes,1),1);
belief_dyns = {@(b, u)beliefDynamicsGMM(b, u,HumanMind(dt),HumanReactionModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel())};
agents{1} = AgentPlattform(dt,horizonSteps,1,belief_dyns);
agents{2} = AgentArm(dt,horizonSteps,2,belief_dyns);
agents{3} = AgentArm(dt,horizonSteps,3,belief_dyns);
agents{4} = AgentArm(dt,horizonSteps,4,belief_dyns);
% agents{2} = AgentBelt(dt,horizonSteps,2);
% agents{3} = AgentCrane(dt,horizonSteps,3);
% agents{4} = AgentCrane(dt,horizonSteps,4);

%% Setup start and goal/target state

u_guess=cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
for i=1:size(interfDiGr.Nodes,1)
    u_guess{i,1} = zeros(agents{1}.total_uDim,horizonSteps-1);
    u_guess{i,2} = zeros(agents{2}.total_uDim,horizonSteps-1);
    u_guess{i,3} = zeros(agents{3}.total_uDim,horizonSteps-1);
    u_guess{i,4} = zeros(agents{4}.total_uDim,horizonSteps-1);
end
% initial guess, less iterations needed if given well
% guess all agents for every agent, 4x4 x uDim x horiz
% u_guess(:,1,1,:)=1.0;
% u_guess(:,1,2,:)=-1.0;
% u_guess(:,2,1,:)=0.0;
% u_guess(:,2,2,:)=0.0;
% u_guess(:,3,1,:)=1.0;
% u_guess(:,3,2,:)=1.0;
% u_guess(:,4,1,:)=-1.0;
% u_guess(:,4,2,:)=-1.0;
b0=cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
% this is a 2D cell because each agent has different format of belief
% states
% each agent holds the belief of other agents, but in a later version,
% this can be limited to neighbors of interference graph
for i=1:size(interfDiGr.Nodes,1)
    %{4x4}x6
    b0{i,1} = [mu_a1;sig_a1(:);weight_a1;mu_a2;sig_a2(:);weight_a2];
    b0{i,2} = [mu_b;sig_b(:)];
    b0{i,3} = [mu_c;sig_c(:)];
    b0{i,4} = [mu_d;sig_d(:)];
end
%????????????????
% true states of agents are 2D position vectors
x_true = zeros(size(interfDiGr.Nodes,1),agents{3}.motionModel.stDim);
% we select component 1 as true goal
x_true(1,:)=mu_a1(3:4);
x_true(2,:)=mu_b;
x_true(3,:)=mu_c;
x_true(4,:)=mu_d;

%% these are old codes remained
Op.plot = -1; % plot the derivatives as well

% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','r','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

%% === run the optimization
for i_sim = 1:simulation_steps
    u = cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
    b = cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
    L_opt = cell(size(interfDiGr.Nodes,1),1);
    cost = cell(size(interfDiGr.Nodes,1),1);
    finished = cell(size(interfDiGr.Nodes,1),1);
    for i = 1:size(interfDiGr.Nodes,1)
        finished{i}= false;
    end
    for iter = 1:20
        if iter == 1
            for i = 1:size(interfDiGr.Nodes,1)
                for j = 1:size(interfDiGr.Nodes,1)
                    u{i,j} = [];
                    b{i,j} = [];
                end
                cost{i} = [];
            end
        end

        for i = 1:size(interfDiGr.Nodes,1)
            if finished{i}~=true
                [bi,ui,cost{i},L_opt{i},~,~, finished{i}] ...
                    = agents{i}.iLQG_one_it...
                    (interfDiGr, {b0{i,:}}, Op, iter,squeeze(u_guess(i,:,:,:)),...
                    u{i},b{i}, cost{i});
                for j=1:size(interfDiGr.Nodes,1)
                    u{i,j} = ui{j};
                    b{i,j} = bi{j};
                end
            end
        end
        % up till now, in b{i} only ith (agent) row are changing, 
        % jth (neighbor agents) have non-zero values but do not
        % change. in u{i} only ith (agent) row are non-zero
        for i = 1:size(interfDiGr.Nodes,1)
            [eid,nid] = inedges(interfDiGr,i);
            for j_nid = 1:length(nid)
                j = nid(j_nid);
%                     b{i}(j,:,:) = b{j}(j,:,:);
% maybe not necessary to exchange b because b will be updated in forward
% pass anyway
%                     u{i}(j,:,:) = u{j}(j,:,:);%simple share eigen-policy
                    % which results in zero est-policy error from the real
%                     u{i}(j,:,:) = (u{j}(j,:,:) + u{i}(j,:,:))/2;%Salehisadaghiani method


                % if you also want to let those agents without direct
                % coupling also learn the policies, use Ye & Hu's
                % update methods. In their case, coupling agents do 
                % not have to be neighbors in communication graph
                % as long as all agents are connected by comm graph
            end
        end
        for ii =1:1
            d_u_est = u;%only to make the size the same, values will not be used
            for i = 1:size(interfDiGr.Nodes,1)
                for j=1:size(interfDiGr.Nodes,1)
                    if i==j
                        d_u_est{i,j}(:,:) = zeros(size(u{i,j},1),horizonSteps-1);
                    else
                        
                        sum_est = zeros(size(u{i,j},1),horizonSteps-1);
                        for k=1:size(interfDiGr.Nodes,1)
%                             if size(u{i,j},1)>2
%                                 u_ij = u{i,j}(5:6,:);
%                             else
                                u_ij = u{i,j};
%                             end
%                             if size(u{k,j},1)>2
%                                 u_kj = u{k,j}(5:6,:);
%                             else
                                u_kj = u{k,j};
%                             end
                            sum_est = sum_est+adjGr(i,k)*(u_ij-u_kj);
                        end
                        d_u_est{i,j}(:,:)=-(sum_est+adjGr(i,j)*(u{i,j}-u{j,j}));
                    end
                end
            end
            for i = 1:size(interfDiGr.Nodes,1)
                for j = 1:size(interfDiGr.Nodes,1)
                    u{i,j} = u{i,j} + 0.2*d_u_est{i,j};
                end
            end
        end

        if finished{1}% && finished{2} && finished{3} && finished{4} 
            break;
        end
%             error_policy_3_from_1 = squeeze(u{3,1}(1,:,:)-u{1,1}(1,:,:));
%             error_policy_4_from_3 = squeeze(u{4,1}(3,:,:)-u{3,1}(3,:,:));
%             figure(3)
%             plot(error_policy_3_from_1(1,:))
%             hold on
%             plot(error_policy_3_from_1(2,:))
%             figure(4)
%             plot(error_policy_4_from_3(1,:))
%             hold on
%             plot(error_policy_4_from_3(2,:))
    end

    for i = 1:size(interfDiGr.Nodes,1)
        % iLQG iteration finished, take the policy to execute
        agents{i}.updatePolicy(b{i},u{i},L_opt{i});
        u_guess(i,:,:,:) = u{i};
        % guess value only used for the first iteration of each MPC iteration
    end


%     assignin('base', 'om', om)
%     assignin('base', 'lims', Op.lims)
%     [b_nom2,u_nom2,L_opt2,Vx2,Vxx2,cost2] = agents{2}.iLQG_GMM(b0{2}, u_guess, Op);
%     agents{2}.updatePolicy(b_nom2,u_nom2,L_opt2);
    if i_sim < 2
        show_mode = EQUAL_WEIGHT_TO_BALL_FEEDBACK;
    else
        show_mode = BALL_WISH_WITHOUT_HUMAN_INPUT;
    end
    time_past = (i_sim-1) * mpc_update_period;
    assignin('base', 'b0', b0)
    assignin('base', 'agents', agents)
    assignin('base', 'update_steps', update_steps)
    assignin('base', 'time_past', time_past)
    assignin('base', 'show_mode', show_mode)
    assignin('base', 'x_true', x_true)
    [~, b0, x_true] = animateHeteroMultiagent(interfDiGr,agents, b0, x_true,update_steps,time_past, show_mode);
%     b0{1}(1:2) = x_true_final(1:2);
end
% figure(3)
% plot(error_policy_3_from_1(1,:),'.k')
% plot(error_policy_3_from_1(2,:),'.k')
% figure(4)
% plot(error_policy_4_from_3(1,:),'.k')
% plot(error_policy_4_from_3(2,:),'.k')
end
