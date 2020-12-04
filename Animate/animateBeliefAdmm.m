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
        t_human_withdraw = 0.7;
        comp_sel =1;
    case BALL_WISH_WITH_OPPOSITE_HUMAN_INPUT
        t_human_withdraw = 0.5;
        comp_sel =2;
    case REST_WISH_WITHOUT_HUMAN_INPUT
        t_human_withdraw = 0.0;
    case REST_WISH_WITH_HUMAN_INPUT
        t_human_withdraw = 1.0;
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
z_save = zeros(2,nSteps-1);
failed = 0;
b = b0;
b_last=b0;
mu_save=cell(size(D.Nodes,1),agents{1}.components_amount);
sig_save=cell(size(D.Nodes,1),agents{1}.components_amount);
for i=1:size(D.Nodes,1)
    for i_comp=1:agents{i}.components_amount
        if i==1
            components_amount=2;
            [x_platf_comp, P_platf, w] = b2xPw(b{1,1}, 4, components_amount);
            mu_save{i,i_comp}(:,1) = x_platf_comp{i_comp};
            sig_save{i,i_comp}(:,1) = P_platf{i_comp}(:);
        else
            
            mu_save{i,i_comp}(:,1) = b{i,i}(1:2);
            sig_save{i,i_comp}(:,1) = b{i,i}(3:6);
        end
    end
end
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
            x_mind = [x_true(5,:)';x_true(1,:)'];
            %1 is plattform positions, 5 is transport band
            z_human_react = agents{1}.obsModel.getObservation(x_mind,'nonoise');
            if k*agents{i}.motionModel.dt>t_human_withdraw
                v_man=[0;0];
                z_human_react(1)=0;
                z_human_react(2)=0;
            else
                v_man=[z_human_react(1)*cos(z_human_react(2));z_human_react(1)*sin(z_human_react(2))]/3;
                z_human_react(1:2)=z_human_react(1:2) + chol(agents{1}.obsModel.R_speed)' * randn(2,1);
            end
            u_plattform_drive=u{i}(5:6);
%             u_plattform_drive=(u{2}+u{3}+u{4})/3;
            if show_mode>6 ||show_mode==3% towards resting place
                x_mind_next = agents{i}.motionModel.evolve(x_mind,[u{i}(3:4);u_plattform_drive+v_man],processNoise);
            else
                x_mind_next = agents{i}.motionModel.evolve(x_mind,[u{i}(1:2);u_plattform_drive+v_man],processNoise);
                %1234 are agent positions, 5 is transport band
                x_true(5,:)=x_mind_next(1:2)';
            end
            x_true(1,:) = x_mind_next(3:4)';
            z{1} = agents{1}.obsModel.getObservation(x_mind_next,'truenoise');
            z{1}(1:2) = z_human_react(1:2);
            z_save(:,k)=z{1}(3:4);
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
        if i==1
            u_man=[0;0;0;0;v_man];
            [b_next_i,mu_i,sig_i,~] = agents{i}.getNextEstimation(b{i,i},u{i}+u_man,z{i});
        
        elseif i<5
%             u_man=[0;0;0;0;v_man];
            [b_next_i,mu_i,sig_i,~] = agents{i}.getNextEstimation(b{i,i}(1:6),u{i}+v_man,z{i});
            
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
%     for i = 1:size(D.Nodes,1)
%         [eid,nid] = inedges(D,i);
%         for j_nid = 1:length(nid)
%             j = nid(j_nid);
%             % I close up the estimation exchange between agents!
%             % but it can be reopened
%             b{i,j} = b{j,j};
%             %if you dont exchange this, MPC will always start from the very
%             %beginning state
% %             u{i}(j,:) = u{j}(j,:);
%         end
%     end
    %% now for save
    for i = 1:size(D.Nodes,1)%agents_amount
        for i_comp = 1 : agents{i}.components_amount
            mu_save{i,i_comp}(:,k+1) = mu{i,i_comp};
            sig_save{i,i_comp}(:,k+1) = sig{i,i_comp}(:);
%             weight_save{i,i_comp}(:,k+1) = weight{i}(i_comp);
        end
    end
    x_save(:,:,k+1) = x_true;
    
    if k>0
% % %         figure(fig_xy)
% % %         plot([x_save(1,1,k-1),x_save(1,1,k)],[x_save(1,2,k-1),x_save(1,2,k)],'-r.')
% % %         hold on
% % % 
% % %         plot(mu_save{1,1}(1,k+1),mu_save{1,1}(2,k+1),'bo')%packet, target
% % % %         hold on
% % %         grid on
% % %         axis equal
% % % %         plot(mu_save{1,1}(3,k+1),mu_save{1,1}(4,k+1),'bo')%packet, plattform
% % %         plot(mu_save{1,2}(1,k+1),mu_save{1,2}(2,k+1),'mo')%rest, target
% % % %         plot(mu_save{1,2}(3,k+1),mu_save{1,2}(4,k+1),'mo')%rest, plattform
% % % %         plot(mu_save{2,1}(1,k+1),mu_save{2,1}(2,k+1),'ro')
% % % %         plot(mu_save{3,1}(1,k+1),mu_save{3,1}(2,k+1),'ko')
% % % %         plot(mu_save{4,1}(1,k+1),mu_save{4,1}(2,k+1),'go')
% % %         plot(z{1}(3),z{1}(4),'b*')
% % %         if draw_cov
% % %             pointsToPlot = drawResultGMM([mu_save{1,1}(:,k+1); sig_save{1,1}(:,k+1)], 4);
% % %     %         pointsToPlot = drawResult([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], 2);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
% % %             pointsToPlot = drawResultGMM([mu_save{1,2}(:,k+1); sig_save{1,2}(:,k+1)], 4);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'m')
% % %             pointsToPlot = drawResult([mu_save{2,1}(:,k+1); sig_save{2,1}(:,k+1)], 2);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
% % %             pointsToPlot = drawResult([mu_save{3,1}(:,k+1); sig_save{3,1}(:,k+1)], 2);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'k')
% % %             pointsToPlot = drawResult([mu_save{4,1}(:,k+1); sig_save{4,1}(:,k+1)], 2);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
% % %             pointsToPlot = drawResult([mu_save{5,1}(:,k+1); sig_save{5,1}(:,k+1)], 2);
% % %             plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
% % %         end
% % %         
% % % %         plot(z{1}(1),z{1}(2),'b*')
% % %         plot(z{2}(1),z{2}(2),'r*')
% % %         plot(z{3}(1),z{3}(2),'k*')
% % %         plot(z{4}(1),z{4}(2),'g*')
% % %         plot(z{5}(1),z{5}(2),'r*')
% % %         
% % %         plot([x_save(2,1,k-1),x_save(2,1,k)],[x_save(2,2,k-1),x_save(2,2,k)],'-k.')
% % %         plot([x_save(3,1,k-1),x_save(3,1,k)],[x_save(3,2,k-1),x_save(3,2,k)],'-r.')
% % %         plot([x_save(4,1,k-1),x_save(4,1,k)],[x_save(4,2,k-1),x_save(4,2,k)],'-r.')
% % %         plot([x_save(5,1,k-1),x_save(5,1,k)],[x_save(5,2,k-1),x_save(5,2,k)],'-r.')
% % %         %     
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

        figure(fig_w)
        plot([agents{1}.motionModel.dt*(k-1),agents{1}.motionModel.dt*(k)],[b_last{1,1}(21),b{1,1}(21)],'-b','Linewidth',2.0)
        hold on
        plot([agents{1}.motionModel.dt*(k-1),agents{1}.motionModel.dt*(k)],[b_last{1,1}(42),b{1,1}(42)],'-r','Linewidth',2.0)

        axis([0,3,0,1])
    
        figure(fig_w+5)
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
        figure(fig_w+1)
        plot([time_past + agents{1}.motionModel.dt*(k-1),time_past + agents{1}.motionModel.dt*(k)],[(u{2}(1)+u{3}(1)+u{4}(1))/3,(u{2}(1)+u{3}(1)+u{4}(1))/3],'b-','Linewidth',2.0)
        hold on
        plot([time_past + agents{1}.motionModel.dt*(k-1),time_past + agents{1}.motionModel.dt*(k)],[(u{2}(2)+u{3}(2)+u{4}(2))/3,(u{2}(2)+u{3}(2)+u{4}(2))/3],'r-','Linewidth',2.0)
        figure(fig_w+2)
%         v_man=[1,2];
        plot([time_past + agents{1}.motionModel.dt*(k-1),time_past + agents{1}.motionModel.dt*(k)],...
            [v_man(1),v_man(1)],'b-','Linewidth',2.0)
        hold on
        plot([time_past + agents{1}.motionModel.dt*(k-1),time_past + agents{1}.motionModel.dt*(k)],...
            [v_man(2),v_man(2)],'r-','Linewidth',2.0)

        figure(fig_w+3)
        plot([time_past + agents{1}.motionModel.dt*(k-1),time_past + agents{1}.motionModel.dt*(k)],[u{1}(2),u{1}(2)],'r-','Linewidth',2.0)
        hold on
    end
    b_last=b;
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
        pause(0.02);
    end
end
figure(fig_xy)
%human/plattform position
plot(squeeze(x_save(1,1,:)),squeeze(x_save(1,2,:)),'m-','Linewidth',2.0)
% plot(mu_save{1,1}(1,:),mu_save{1,1}(2,:),'bo')
hold on
grid on
axis equal
plot(mu_save{1,1}(1,:),mu_save{1,1}(2,:),'b*')
plot(mu_save{1,2}(1,:),mu_save{1,2}(2,:),'r*')

plot(z_save(1,:),z_save(2,:),'m+')
%         plot(mu_save{1,2}(3,k),mu_save{1,2}(4,k),'mo')
plot(mu_save{2,1}(1,:),mu_save{2,1}(2,:),'-k','Linewidth',2.0)
plot(mu_save{3,1}(1,:),mu_save{3,1}(2,:),'-k','Linewidth',2.0)
plot(mu_save{4,1}(1,:),mu_save{4,1}(2,:),'-k','Linewidth',2.0)
plot(mu_save{5,1}(1,:),mu_save{5,1}(2,:),'-k','Linewidth',2.0)
for k=1:nSteps
    pointsToPlot = drawResultGMM([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], 4);
    plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
    pointsToPlot = drawResultGMM([mu_save{1,2}(:,k); sig_save{1,2}(:,k)], 4);
    plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
end

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
%     plot(squeeze(x_save(1,1,:)),squeeze(x_save(1,2,:)),'-r.')
%     plot(squeeze(x_save(2,1,:)),squeeze(x_save(2,2,:)),'-k.')
%     plot(squeeze(x_save(3,1,:)),squeeze(x_save(3,2,:)),'-r.')
%     plot(squeeze(x_save(4,1,:)),squeeze(x_save(4,2,:)),'-r.')
%     plot(squeeze(x_save(5,1,:)),squeeze(x_save(5,2,:)),'-r.')
    figure(fig_w+5)
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
    
    figure(fig_w)
    title('Gewicht der Wünsche')
    xlabel('t(s)')
    ylabel('Gewicht')
    legend('A','B')
    hold off
    
    figure(fig_w+1)
%     subplot(2,2,1)
    axis([0,3,-5,5])
    title('Unterstützung Plattform aus Summe der Assistenten')
    xlabel('t(s)')
    ylabel('vel(m/s)')
    legend('x','y')
%     legend('x','y','Band')
    grid
    hold off
    figure(fig_w+2)
    axis([0,3,-3,3])
    title('Mensch selbst')
    xlabel('t(s)')
    ylabel('vel(m/s)')
    legend('x','y')
    grid
    hold off
    figure(fig_w+3)
    axis([0,3,-3,3])
    title('Bewegung des Ziels A')
    xlabel('t(s)')
    ylabel('vel(m/s)')
    legend('y')
    grid
    hold off
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
[x_m,y_m] = meshgrid(2:0.5:11,-2:0.5:5);
X=2:0.5:11;
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
surf(x_m,y_m,Z-max(max(Z))-0.5),shading flat
    xlabel('x(m)')
    ylabel('y(m)')
    grid on
    axis([3,11,-2,5])
legend('wahre Plattform','Ziel A','Ziel B','Messung Plattform','Assistenten')
end