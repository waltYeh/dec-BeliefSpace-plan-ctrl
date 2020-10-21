function [failed, b_f, x_true_final]...
    = animateBeliefAdmm(fig_xy,fig_w,D,agents, b0, x_true, nSteps,time_past, show_mode,draw_cov)
% longer, clear wish, shorter, less overshoot
t_human_withdraw = 0.5;
comp_sel =1;
use_bad_man_speed = true;
EQUAL_WEIGHT_BALANCING = 1;
EQUAL_WEIGHT_TO_BALL_FEEDBACK = 2;
EQUAL_WEIGHT_TO_REST_FEEDBACK = 3;
BALL_WISH_WITHOUT_HUMAN_INPUT = 4;
BALL_WISH_WITH_HUMAN_INPUT = 5;
BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT = 6;
REST_WISH_WITHOUT_HUMAN_INPUT = 7;
REST_WISH_WITH_HUMAN_INPUT = 8;
REST_WISH_WITH_OPPOSITE_HUMAN_INPUT = 9;
CHANGE_WISHES = 10;
% show_mode = EQUAL_WEIGHT_BALANCING;
switch show_mode
    case EQUAL_WEIGHT_BALANCING
        t_human_withdraw = 0.0;
    case EQUAL_WEIGHT_TO_BALL_FEEDBACK
        t_human_withdraw = 0.2;
        comp_sel =1;
    case EQUAL_WEIGHT_TO_REST_FEEDBACK
        t_human_withdraw = 0.3;
        comp_sel =2;        
    case BALL_WISH_WITHOUT_HUMAN_INPUT
        t_human_withdraw = 0.0;
    case BALL_WISH_WITH_HUMAN_INPUT
        t_human_withdraw = 0.2;
        comp_sel =1;
    case BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT
        t_human_withdraw = 0.5;
        comp_sel =2;
    case REST_WISH_WITHOUT_HUMAN_INPUT
        t_human_withdraw = 0.0;
    case REST_WISH_WITH_HUMAN_INPUT
        t_human_withdraw = 0.7;
        comp_sel =2;
    case REST_WISH_WITH_OPPOSITE_HUMAN_INPUT
        t_human_withdraw = 0.5;
        comp_sel =1;
    case CHANGE_WISHES
        t_human_withdraw = 5;
        comp_sel =1;
        use_bad_man_speed = true;
end
% component_stDim = agents{1}.motionModel.stDim;
% component_bDim = component_stDim + component_stDim^2 + 1;
% shared_uDim = 2;
% component_alone_uDim = agents{1}.motionModel.ctDim - shared_uDim;
% 
% components_amount = length(b0)/component_bDim;
%     u_man = [u(end-shared_uDim+1);u(end)]
    
% stDim = motionModel.stDim;

agents_amount = length(agents);
max_components_amount=agents{1}.components_amount;%max(agents{1}.components_amount, agents{2}.components_amount);
x_save = x_true;

failed = 0;
b = b0;

for k = 1:nSteps-1
    u=cell(5,1);
    z=cell(5,1);
    for i=1:size(D.Nodes,1)
        u{i} = agents{i}.getNextControl(b(i,:));
    end
    %% update physical part
    for i=1:size(D.Nodes,1)  % the real states of four robots
        processNoise = agents{i}.motionModel.generateProcessNoise(x_true(i,:),u{i});
        if i==1
            if show_mode>6 ||show_mode==3% towards resting place
                x_mind = [x_true(5,:)';x_true(1,:)'];
                x_mind_next = agents{i}.motionModel.evolve(x_mind,[u{i}(3:4);u{i}(5:6)],processNoise);
                x_true(1,:) = x_mind_next(3:4)';
                z{i} = agents{i}.obsModel.getObservation(x_mind_next,'truenoise');
            else
                x_mind = [x_true(5,:)';x_true(1,:)'];
                x_mind_next = agents{i}.motionModel.evolve(x_mind,[u{i}(1:2);u{i}(5:6)],processNoise);
                x_true(1,:) = x_mind_next(3:4)';
                %1234 are agent positions, 5 is transport band
                x_true(5,:)=x_mind_next(1:2)';
                z{i} = agents{i}.obsModel.getObservation(x_mind_next,'truenoise');
            end
            if k*agents{i}.motionModel.dt>t_human_withdraw
                z{i}(1:2)=[0;0];
            else
%                 z_human_react(1:2)=[speed_man;direction_man] + chol(obsModel.R_speed)' * randn(2,1);
            end
%         elseif i==2
% %             x_true_x_last = x_true(i,1);
% %             x_true(i,:) = agents{i}.motionModel.evolve(x_true(i,:)',u{i},processNoise);
% %             x_true(i,1)=x_true_x_last;
%             z{i} = agents{i}.obsModel.getObservation(x_true(i,:),'truenoise');

        elseif i<5
            %% manipulated
            [x1, P1, w1] = b2xPw(b{1,1}, 4, 2);
            K_feedback = agents{1}.L_opt(5:6,[21,42],k);
%             u_w = zeros(2,1);
%             for hori=1:size(K_feedback,3)
            u_w = K_feedback*w1;
%             u{i} = u{i}+u_w;
            x_true(i,:) = agents{i}.motionModel.evolve(x_true(i,:)',u{i},processNoise);
            z{i} = agents{i}.obsModel.getObservation(x_true(i,:),'truenoise');
            
        elseif i==5
            x_true(6,:) = agents{i}.motionModel.evolve(x_true(6,:)',u{i},processNoise);
            z{i} = agents{i}.obsModel.getObservation(x_true(6,:),'truenoise');  
        end
    end
    %% now do the machine part
    for i = 1:size(D.Nodes,1)%agents_amount
        % b_next already contains all info of mu_i and sig_i
        if i<5 && i>1
            [b_next_i,mu_i,sig_i,~] = agents{i}.getNextEstimation(b{i,i}(1:6),u{i},z{i});
            components_amount=2;
            [x_platf_comp, P_platf, w] = b2xPw(b{1,1}, 4, components_amount);

            x_platf_weighted = zeros(2,components_amount);
            for ii=1:components_amount
                x_platf_weighted(:,ii)=transpose(x_platf_comp{ii}(3:4)*w(ii));
            end
            x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];
%             b_next_i = [b_next_i;x_platf];
        else
            [b_next_i,mu_i,sig_i,~] = agents{i}.getNextEstimation(b{i,i},u{i},z{i});
        end
        %should pass in the control and measurement of connected agents in
        %order to do estimations for them!
        b{i,i}=b_next_i;
%         b{2}(i,:)=b_next_i;
%         b{3}(i,:)=b_next_i;
%         b{4}(i,:)=b_next_i;
        for i_comp=1:agents{i}.components_amount % only one component
            mu{i,i_comp}=mu_i{i_comp};
            sig{i,i_comp}=sig_i{i_comp};
%             weight{i}(i_comp)=weight_i(i_comp);
        end
    end
    for i = 1:size(D.Nodes,1)
        [eid,nid] = inedges(D,i);
        for j_nid = 1:length(nid)
            j = nid(j_nid);
            % I close up the estimation exchange between agents!
            % but it can be reopened
            b{i,j} = b{j,j};
            %if you dont exchange this, MPC will always start from the very
            %beginning state
%             u{i}(j,:) = u{j}(j,:);
        end
    end
    %% now for save
    for i = 1:size(D.Nodes,1)%agents_amount
        for i_comp = 1 : agents{i}.components_amount
            mu_save{i,i_comp}(:,k+1) = mu{i,i_comp};
            sig_save{i,i_comp}(:,k+1) = sig{i,i_comp}(:);
%             weight_save{i,i_comp}(:,k+1) = weight{i}(i_comp);
        end
    end
    x_save(:,:,k+1) = x_true;
    
    if k>1
        figure(fig_xy)
        plot([x_save(1,1,k-1),x_save(1,1,k)],[x_save(1,2,k-1),x_save(1,2,k)],'-r.')
        hold on

        plot(mu_save{1,1}(1,k+1),mu_save{1,1}(2,k+1),'bo')%packet, target
%         hold on
        grid on
        axis equal
%         plot(mu_save{1,1}(3,k+1),mu_save{1,1}(4,k+1),'bo')%packet, plattform
        plot(mu_save{1,2}(1,k+1),mu_save{1,2}(2,k+1),'mo')%rest, target
%         plot(mu_save{1,2}(3,k+1),mu_save{1,2}(4,k+1),'mo')%rest, plattform
%         plot(mu_save{2,1}(1,k+1),mu_save{2,1}(2,k+1),'ro')
%         plot(mu_save{3,1}(1,k+1),mu_save{3,1}(2,k+1),'ko')
%         plot(mu_save{4,1}(1,k+1),mu_save{4,1}(2,k+1),'go')
        plot(z{1}(3),z{1}(4),'b*')
        if draw_cov
            pointsToPlot = drawResultGMM([mu_save{1,1}(:,k+1); sig_save{1,1}(:,k+1)], 4);
    %         pointsToPlot = drawResult([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], 2);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
            pointsToPlot = drawResultGMM([mu_save{1,2}(:,k+1); sig_save{1,2}(:,k+1)], 4);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'m')
            pointsToPlot = drawResult([mu_save{2,1}(:,k+1); sig_save{2,1}(:,k+1)], 2);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
            pointsToPlot = drawResult([mu_save{3,1}(:,k+1); sig_save{3,1}(:,k+1)], 2);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'k')
            pointsToPlot = drawResult([mu_save{4,1}(:,k+1); sig_save{4,1}(:,k+1)], 2);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
            pointsToPlot = drawResult([mu_save{5,1}(:,k+1); sig_save{5,1}(:,k+1)], 2);
            plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
        end
        
%         plot(z{1}(1),z{1}(2),'b*')
        plot(z{2}(1),z{2}(2),'r*')
        plot(z{3}(1),z{3}(2),'k*')
        plot(z{4}(1),z{4}(2),'g*')
        plot(z{5}(1),z{5}(2),'r*')
        
        plot([x_save(2,1,k-1),x_save(2,1,k)],[x_save(2,2,k-1),x_save(2,2,k)],'-k.')
        plot([x_save(3,1,k-1),x_save(3,1,k)],[x_save(3,2,k-1),x_save(3,2,k)],'-r.')
        plot([x_save(4,1,k-1),x_save(4,1,k)],[x_save(4,2,k-1),x_save(4,2,k)],'-r.')
        plot([x_save(5,1,k-1),x_save(5,1,k)],[x_save(5,2,k-1),x_save(5,2,k)],'-r.')
        %     
        %     pointsToPlot = drawResultGMM([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], agents{1}.motionModel.stDim);
        %     plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
        %     pointsToPlot = drawResultGMM([mu_save{1,2}(:,k); sig_save{1,2}(:,k)], agents{1}.motionModel.stDim);
        %     plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
%             figure(55)
%             subplot(2,2,1)
%     %         plot(time_past + 0.05*(k-1),u{1}(5),'b.',time_past + 0.05*(k-1),u{1}(6),'r.')
%             plot(time_past + 0.05*(k-1),u{1}(1),'b.',time_past + 0.05*(k-1),u{1}(2),'r.')
%             hold on
%             subplot(2,2,2)
%             plot(time_past + 0.05*(k-1),u{2}(1),'b.',time_past + 0.05*(k-1),u{2}(2),'r.')
%             hold on
%             subplot(2,2,3)
%             plot(time_past + 0.05*(k-1),u{3}(1),'b.',time_past + 0.05*(k-1),u{3}(2),'r.')
%             hold on
%             subplot(2,2,4)
%             plot(time_past + 0.05*(k-1),u{4}(1),'b.',time_past + 0.05*(k-1),u{4}(2),'r.')
%             hold on
        figure(fig_w+1)
        subplot(2,2,1)
%         plot(time_past + agents{1}.motionModel.dt*(k-1),u{1}(5),'b.',time_past + agents{1}.motionModel.dt*(k-1),u{1}(6),'r.')
        plot(time_past + agents{1}.motionModel.dt*(k-1),(u{2}(1)+u{3}(1)+u{4}(1))/3,'b.',time_past + agents{1}.motionModel.dt*(k-1),(u{2}(2)+u{3}(2)+u{4}(2))/3,'r.')
        hold on
        plot(time_past + agents{1}.motionModel.dt*(k-1),u{1}(2),'r+')
        subplot(2,2,2)
        plot(time_past + agents{1}.motionModel.dt*(k-1),u{2}(1),'b.',time_past + agents{1}.motionModel.dt*(k-1),u{2}(2),'r.')
        hold on
        subplot(2,2,3)
        plot(time_past + agents{1}.motionModel.dt*(k-1),u{3}(1),'b.',time_past + agents{1}.motionModel.dt*(k-1),u{3}(2),'r.')
        hold on
        subplot(2,2,4)
        plot(time_past + agents{1}.motionModel.dt*(k-1),u{4}(1),'b.',time_past + agents{1}.motionModel.dt*(k-1),u{4}(2),'r.')
        hold on
    end
    % figure(6)
    % %     
    %     plot([time_past + agents{1}.dt*(k-1),time_past + agents{1}.dt*(k)],[weight_save{1,1}(k),weight_save{1,1}(k+1)],'-ob',[time_past + agents{1}.dt*(k-1),time_past + agents{1}.dt*(k)],[weight_save{1,2}(k),weight_save{1,2}(k+1)],'-ok')
    %     hold on
    % %     time_line = 0:dt:dt*(nSteps);
    %     figure(10)
    %     subplot(2,2,1)
    %     plot(time_past + agents{1}.dt*(k-1),u(5),'b.',time_past + agents{1}.dt*(k-1),u(6),'r.')
    %     hold on
    %     subplot(2,2,2)
    %     plot(time_past + agents{1}.dt*(k-1),v_man(1),'b.',time_past + agents{1}.dt*(k-1),v_man(2),'r.')
    %     hold on
    %     subplot(2,2,3)
    %     plot(time_past + agents{1}.dt*(k-1),u(1),'b.',time_past + agents{1}.dt*(k-1),u(2),'r.')
    %     hold on
    %     subplot(2,2,4)
    %     plot(time_past + agents{1}.dt*(k-1),u(3),'b.',time_past + agents{1}.dt*(k-1),u(4),'r.')
    %     hold on
    if draw_cov
        pause(0.2);
    end
end
figure(fig_xy)

plot(mu_save{1,1}(1,:),mu_save{1,1}(2,:),'bo')
hold on
grid on
axis equal
plot(mu_save{1,1}(1,:),mu_save{1,1}(2,:),'bo')
%         plot(mu_save{1,2}(1,k),mu_save{1,2}(2,k),'mo')
%         plot(mu_save{1,2}(3,k),mu_save{1,2}(4,k),'mo')
plot(mu_save{2,1}(1,:),mu_save{2,1}(2,:),'ro')
plot(mu_save{3,1}(1,:),mu_save{3,1}(2,:),'ko')
plot(mu_save{4,1}(1,:),mu_save{4,1}(2,:),'go')
plot(mu_save{5,1}(1,:),mu_save{5,1}(2,:),'go')
%         pointsToPlot = drawResultGMM([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], 4);
% pointsToPlot = drawResult([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], 2);
% plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
% %         pointsToPlot = drawResultGMM([mu_save{1,2}(:,k); sig_save{1,2}(:,k)], 4);
% %         plot(pointsToPlot(1,:),pointsToPlot(2,:),'m')
% pointsToPlot = drawResult([mu_save{2,1}(:,k); sig_save{2,1}(:,k)], 2);
% plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
% pointsToPlot = drawResult([mu_save{3,1}(:,k); sig_save{3,1}(:,k)], 2);
% plot(pointsToPlot(1,:),pointsToPlot(2,:),'k')
% pointsToPlot = drawResult([mu_save{4,1}(:,k); sig_save{4,1}(:,k)], 2);
% plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
%         plot(z{1}(3),z{1}(4),'b*')
% plot(z{1}(1),z{1}(2),'b*')
% plot(z{2}(1),z{2}(2),'r*')
% plot(z{3}(1),z{3}(2),'k*')
% plot(z{4}(1),z{4}(2),'g*')
% if k>1
    plot(squeeze(x_save(1,1,:)),squeeze(x_save(1,2,:)),'-r.')
    plot(squeeze(x_save(2,1,:)),squeeze(x_save(2,2,:)),'-k.')
    plot(squeeze(x_save(3,1,:)),squeeze(x_save(3,2,:)),'-r.')
    plot(squeeze(x_save(4,1,:)),squeeze(x_save(4,2,:)),'-r.')
    plot(squeeze(x_save(5,1,:)),squeeze(x_save(5,2,:)),'-r.')
    figure(fig_w+1)
    subplot(2,2,1)
    title('Unterstützung Plattform und Förderband')
    xlabel('t(s)')
    ylabel('vel(m/s)')
%     legend('x','y','Band')
    grid
    % hold off
    subplot(2,2,2)
    title('Assistent 2')
    xlabel('t(s)')
    ylabel('vel(m/s)')
%     legend('x','y')
    grid
    % hold off
    subplot(2,2,3)
    title('Assistent 3')
    xlabel('t(s)')
    ylabel('vel(m/s)')
%     legend('x','y')
    grid
    % hold off
    subplot(2,2,4)
    title('Assistent 4')
    xlabel('t(s)')
    ylabel('vel(m/s)')
%     legend('x','y')
    grid
% end
b_f = b;
% if time_past<0.01
%     figure(10)
%     subplot(2,2,1)
%     title('Unterstützung Plattform')
%     xlabel('t(s)')
%     ylabel('vel(m/s)')
%     grid
%     subplot(2,2,2)
%     title('Mensch selbst')
%     xlabel('t(s)')
%     ylabel('vel(m/s)')
%     grid
%     subplot(2,2,3)
%     title('Bewegung des Ziels A')
%     xlabel('t(s)')
%     ylabel('vel(m/s)')
%     grid
%     subplot(2,2,4)
%     title('Bewegung des Ziels B')
%     xlabel('t(s)')
%     ylabel('vel(m/s)')
%     grid
% 
%     figure(5)
%     title('Bewegungen von Plattform, Ziel A und Ziel B')
%     xlabel('x(m)')
%     ylabel('y(m)')
%     figure(6)
%     title('Gewicht der Wünsche')
%     xlabel('t(s)')
%     ylabel('Gewicht')
% end
x_true_final = x_true;
[x_m,y_m] = meshgrid(1:0.5:10,-2:0.5:5);
X=1:0.5:10;
Y=-2:0.5:5;
Z = zeros(size(x_m,1),size(x_m,2));
for i=1:length(X)
    for j=1:length(Y)
        ZZ = agents{1}.obsModel.getObservationNoiseJacobian([0;0;X(i);Y(j)]);
        Z(j,i) = ZZ(end);
%         1/(1/norm([X(i);Y(j)]-[7;0])^2 + 1/norm([X(i);Y(j)]-[3;1])^2 + 1);
    end
end
figure(fig_xy)
surf(x_m,y_m,Z-max(max(Z))-0.5)
% legend('wahre Plattform','Ziel A','Ziel B','Messung der Plattform','Belief A','Belief B')
end