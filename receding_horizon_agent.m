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
mu_1 = [-0.0, 1.3]';
mu_2 = [-1.0, 0.1]';
mu_3 = [1, 0.0]';
mu_4 = [0.0, -0.3]';
sig_1 = diag([0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01]);
sig_3 = diag([0.01, 0.01]);%sigma
sig_4 = diag([0.01, 0.01]);
% weight_1 = 0.9;
% weight_2 = 0.1;
dt = 0.05;
horizon = 2.0;
mpc_update_period = 2;
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
nom_formation_1=[0.5,0.5;
    -0.5,-0.5;
    -0.5,-0.5;
    -1,-1;
    -0.5,-0.5;
    ];%-- formation
nom_formation_2=[-1,1;
    -2,0;
    1,-1;
    -1,-1;
];%z formation
rii_control = [0.8;0.8;0.8;0.8];
incoming_edges = zeros(4,4);
EdgeTable = table([sd' td'],nom_formation_2,'VariableNames',{'EndNodes' 'nom_formation_2' });

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

comm_sd = [1 3 2 3];
comm_td = [2 1 4 4];
commGr = graph(comm_sd,comm_td);
adjGr = full(adjacency(commGr));

agents = cell(size(interfDiGr.Nodes,1),1);
belief_dyns = {@(b, u)beliefDynamicsSimpleAgent(b, u,TwoDPointRobot(dt),TwoDSimpleObsModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel()); 
    @(b, u)beliefDynamicsSimpleAgent(b, u, TwoDPointRobot(dt),TwoDSimpleObsModel())};
agents{1} = AgentArm(dt,horizonSteps,1,belief_dyns);
agents{2} = AgentArm(dt,horizonSteps,2,belief_dyns);
agents{3} = AgentArm(dt,horizonSteps,3,belief_dyns);
agents{4} = AgentArm(dt,horizonSteps,4,belief_dyns);

%% Setup start and goal/target state

u_guess=cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
u_lambda = cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));
uC = cell(size(interfDiGr.Nodes,1),1);
for i=1:size(interfDiGr.Nodes,1)
    u_lambda{i,1} = zeros(agents{1}.total_uDim,horizonSteps-1);
    u_lambda{i,2} = zeros(agents{2}.total_uDim,horizonSteps-1);
    u_lambda{i,3} = zeros(agents{3}.total_uDim,horizonSteps-1);
    u_lambda{i,4} = zeros(agents{4}.total_uDim,horizonSteps-1);
    
    u_guess{i,1} = zeros(agents{1}.total_uDim,horizonSteps-1);
    u_guess{i,1}(1,:) = -0.15;
    u_guess{i,2} = zeros(agents{2}.total_uDim,horizonSteps-1);
    u_guess{i,2}(1,:) = 0.1;
    u_guess{i,3} = zeros(agents{3}.total_uDim,horizonSteps-1);
    u_guess{i,3}(1,:) = -1.0;
    u_guess{i,4} = zeros(agents{4}.total_uDim,horizonSteps-1);
    uC{i} = u_guess{i,i};
end% initial guess, less iterations needed if given well
% guess all agents for every agent, 4x4x2x40
% u_guess(:,1,1,:)=1.0;
% u_guess(:,1,2,:)=-1.0;
% u_guess(:,2,1,:)=0.0;
% u_guess(:,2,2,:)=0.0;
% u_guess(:,3,1,:)=1.0;
% u_guess(:,3,2,:)=1.0;
% u_guess(:,4,1,:)=-1.0;
% u_guess(:,4,2,:)=-1.0;
b0=cell(size(interfDiGr.Nodes,1),size(interfDiGr.Nodes,1));

% each agent holds the belief of other agents, but in a later version,
% this will be limited to neighbors

for i=1:size(interfDiGr.Nodes,1)
    %{4x4}x6
    b0{i,1} = [mu_1;sig_1(:)];
    b0{i,2} = [mu_2;sig_2(:)];
    b0{i,3} = [mu_3;sig_3(:)];
    b0{i,4} = [mu_4;sig_4(:)];
end
x_true = zeros(size(interfDiGr.Nodes,1),agents{1}.motionModel.stDim);
x_true(1,:)=mu_1;
x_true(2,:)=mu_2;
x_true(3,:)=mu_3;
x_true(4,:)=mu_4;
% b0={[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2]};

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
    for iter = 1:15
        if iter == 1
            for i = 1:size(interfDiGr.Nodes,1)
                u{i} = [];
                b{i} = [];
                cost{i} = [];
            end
        end

        for i = 1:size(interfDiGr.Nodes,1)
            if finished{i}~=true
                uC_lambda = uC;
                for j=1:size(interfDiGr.Nodes,1)
                    uC_lambda{j} = uC{j}-u_lambda{i,j};
                end
                %% 
                [bi,ui,cost{i},L_opt{i},~,~, finished{i}] ...
                    = agents{i}.iLQG_one_it...
                    (interfDiGr, b0(i,:), Op,  iter,u_guess(i,:),...
                    uC_lambda,uC,b(i,:), cost{i});
                %% 
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
        for i = 1:4
            uC{i} = u{i,i};
        end
        alpha_u = 1.0;
        for i = 1:4
            for j = 1:4
                u_lambda{i,j} = u_lambda{i,j} + alpha_u / ...
                    agents{i}.rho(2) * (u{i,j}-uC{j});
            end
        end
%         if 0
%             for i = 1:4
%                 for j=1:4
%                     if i==j
% %                             d_u_est{i,1}(j,:,:) = zeros(1,2,horizonSteps-1);
%                     else
%                         u{i}(j,:,:) = u{j}(j,:,:);
% %                              u{i}(j,:,:) = 0.3*u{j}(j,:,:) + 0.7*u{i}(j,:,:);
%                     end
%                 end
%             end
%         else
%             for ii =1:1
%                 d_u_est = u;%only to make the size the same, values will not be used
%                 for i = 1:size(interfDiGr.Nodes,1)
%                     for j=1:size(interfDiGr.Nodes,1)
%                         if i==j
%                             d_u_est{i,j}(:,:) = zeros(size(u{i,j},1),horizonSteps-1);
%                         else
%                             sum_est = zeros(size(u{i,j},1),horizonSteps-1);
%                             for k=1:size(interfDiGr.Nodes,1)
%                                 u_ij = u{i,j};
%                                 u_kj = u{k,j};
%                                 sum_est = sum_est+adjGr(i,k)*(u_ij-u_kj);
%                             end
%                             d_u_est{i,j}(:,:)=-(sum_est+adjGr(i,j)*(u{i,j}-u{j,j}));
%                         end
%                     end
%                 end
%                 for i = 1:size(interfDiGr.Nodes,1)
%                     for j = 1:size(interfDiGr.Nodes,1)
%                         u{i,j} = u{i,j} + 0.4*d_u_est{i,j};
%                     end
%                 end
%             end
%         end

%             error_policy_3_from_1 = squeeze(u{3,1}(1,:,:)-u{1,1}(1,:,:));
%             error_policy_4_from_3 = squeeze(u{4,1}(3,:,:)-u{3,1}(3,:,:));
%             figure(3)
%             plot(error_policy_3_from_1(1,:))
%             hold on
%             plot(error_policy_3_from_1(2,:))
%             figure(4)
%             plot(error_policy_4_from_3(1,:))
%             hold on
% %             plot(error_policy_4_from_3(2,:))
        if 1
            figure(iter)
            for agent_i=1:4
                subplot(2,2,agent_i)
                for agent_j=1:4
                    
                    if agent_i == agent_j
                        pattern = '.';
                    else
                        pattern = '-';
                    end
                    if agent_i ~= agent_j&&finished{agent_j}
                        % those agents who has finished do not update their
                        % estimation about agent i any more
                    else
                        try
                            plot(1:horizonSteps-1,squeeze(u{agent_j,agent_i}(1,:)),pattern)
                            hold on
                            plot(1:horizonSteps-1,squeeze(u{agent_j,agent_i}(2,:)),pattern)
                            hold on
                        catch ME
                            keyboard
                        end
                    end
                end
                title(strcat('Policy of agent ',num2str(agent_i)))
            end
        end
        
        for i = 1:size(interfDiGr.Nodes,1)
            % iLQG iteration finished, take the policy to execute
            agents{i}.updatePolicy(b(i,:),u(i,:),L_opt{i});
            % guess value only used for the first iteration of each MPC iteration
        end% guess value only used for the first iteration of each MPC iteration
        if finished{1} && finished{2} && finished{3} && finished{4} 
            break;
        end
        time_past = (i_sim-1) * mpc_update_period;
        draw_ball=false;
        [~, ~, ~] = animateHeteroMultiagent(interfDiGr,agents, b0, x_true,update_steps,time_past, show_mode,draw_ball);

    end% end of iLQG iterations


    for i = 1:size(interfDiGr.Nodes,1)
        % iLQG iteration finished, take the policy to execute
        u_guess(i,:) = u(i,:);
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
    assignin('base', 'interfDiGr', interfDiGr)
    assignin('base', 'b0', b0)
    assignin('base', 'agents', agents)
    assignin('base', 'update_steps', update_steps)
    assignin('base', 'time_past', time_past)
    assignin('base', 'show_mode', show_mode)
    assignin('base', 'x_true', x_true)
    for i = 1:size(interfDiGr.Nodes,1)
        agents{i}.ctrl_ptr = 1;
    end
    draw_ball=true;
    [~, b0_next, x_true_next] = animateHeteroMultiagent(interfDiGr,agents, b0, x_true,update_steps,time_past, show_mode,draw_ball);
%     b0{1}(1:2) = x_true_final(1:2);
    b0 = b0_next;
    x_true = x_true_next;
%     b0{1}(1:2) = x_true_final(1:2);
end
% figure(3)
% plot(error_policy_3_from_1(1,:),'.k')
% plot(error_policy_3_from_1(2,:),'.k')
% figure(4)
% plot(error_policy_4_from_3(1,:),'.k')
% plot(error_policy_4_from_3(2,:),'.k')
end
