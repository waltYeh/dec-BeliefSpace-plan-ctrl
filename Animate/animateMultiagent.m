function [failed, b_f, x_true_final] = animateMultiagent(agents, b0, x_true, nSteps,time_past, show_mode)
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

% mu = cell(components_amount,1);
% sig = cell(components_amount,1);
% weight = zeros(components_amount,1);



% mu_save = zeros(agents_amount,max_components_amount,agents{1}.component_stDim,nSteps);
% sig_save = zeros(agents_amount,max_components_amount,agents{1}.component_stDim^2,nSteps);
% % weight_save = zeros(agents_amount,max_components_amount,nSteps);
% x_save = [];
% x_true = zeros(agents_amount,agents{1}.component_stDim);
% mu = zeros(agents_amount,max_components_amount,agents{1}.component_stDim);
% sig = zeros(agents_amount,max_components_amount,agents{1}.component_stDim^2);
% % weight = cell(agents_amount,1);
% for i = 1:agents_amount
%     [mu_i, sig_i] = b2xP(b0{i}, agents{i}.component_stDim);
% %     [mu_i,sig_i,weight_i] = b2xPw(b0(i,:), agents{i}.component_stDim, agents{i}.components_amount);
%     for i_comp=1:agents{i}.components_amount
%         mu(i,i_comp,:)=mu_i;%{i_comp};
%         sig(i,i_comp,:)=sig_i(:);%{i_comp};
% %         weight{i}(i_comp)=weight_i(i_comp);
%     end
% end
% % for i_comp=1:agents{1}.components_amount
% % %     b0_comp = b0((i_comp-1)*component_bDim+1:i_comp*component_bDim);
% % %     mu{i_comp} = b0_comp(1:component_stDim);
% % %     for d = 1:motionModel.stDim
% % %         sig{i_comp}(:,d) = b0_comp(d*component_stDim+1:(d+1)*component_stDim, 1);
% % %     end
% % %     weight(i_comp) = b0_comp(end);
% %     if comp_sel == i_comp
% for i=1:agents_amount
%         x_true = mu;%{1,i_comp};% + chol(sig{i_comp})' * randn(component_stDim,1);
        x_save = x_true;






% for i = 1:agents_amount
%     for i_comp=1:agents{1}.components_amount
%         mu_save(i,i_comp,:,1) = mu(i,i_comp,:);
%         sig_save(i,i_comp,:,1) = sig(i,i_comp,:);
% %         weight_save{i,i_comp} = weight{i}(i_comp);
%     end
% end

failed = 0;
b = b0;
for k = 1:nSteps-1

%     b = zeros(component_bDim*components_amount,1); % current belief
%     for i_comp=1:components_amount
%         b((i_comp-1)*component_bDim+1:(i_comp-1)*component_bDim+component_stDim)=mu{i_comp};
%         b((i_comp-1)*component_bDim+component_stDim+1:(i_comp-1)*component_bDim+component_stDim+component_stDim*component_stDim)=sig{i_comp};
%         b((i_comp)*component_bDim)=weight(i_comp);
%     end
%     b = xPw2b(mu, sig, weight, component_stDim, components_amount);
    u=cell(4,1);
    z=cell(4,1);
    for i=1:4
        u{i} = agents{i}.getNextControl(b{i});
%     u{2} = agents{2}.getNextControl(b{2});
%     u{3} = agents{3}.getNextControl(b{3});
%     u{4} = agents{4}.getNextControl(b{4});
    end
    %% update physical part
    
%     zeroProcessNoise = zeros(2,1);
    % here we only use the input of comp_sel, drop the other target
    % u_for_true includes one target and the assist
    
%     u_for_true = [u((comp_sel-1)*max_components_amount + 1:comp_sel*max_components_amount);u(end-1:end)];
    for i=1:4  
        processNoise = agents{i}.motionModel.generateProcessNoise(x_true(i,:),u{i});
        x_true(i,:) = agents{i}.motionModel.evolve(x_true(i,:)',u{i},processNoise);
        z{i} = agents{i}.obsModel.getObservation(x_true(i,:),'truenoise');
    end
%         
%     good_man_for_ball_should_output = agents{1}.obsModel.getObservation(x_true,'nonoise');
%     good_man_speed_angle=good_man_for_ball_should_output(1:2);
%     v_man = [good_man_speed_angle(1)*cos(good_man_speed_angle(2));
%                 good_man_speed_angle(1)*sin(good_man_speed_angle(2))];
%     
%     if use_bad_man_speed
%         if comp_sel ==1
%             v_man=[0.7;-0.2]*0.94^(k*agents{1}.dt*20)*6;
%             if k*agents{1}.dt>1
%                 v_man=[0.15;0.5]*0.94^((k-nSteps/6)*agents{1}.dt*20)*6;
%             end
%             if k*agents{1}.dt>2.6
%                 v_man=[-0.9;0.1]*0.94^((k-nSteps/3)*agents{1}.dt*20)*6;
%             end
%             if k*agents{1}.dt>4.5
%                 v_man=[0.45;-1.4]*0.94^((k-nSteps/3*2)*agents{1}.dt*20)*6;
%             end
%         elseif comp_sel ==2
%             v_man = [-1.1;1.]*0.94^(k*agents{1}.dt*20)*0.3;
%         end
%         if k*agents{1}.dt>t_human_withdraw
%             v_man=[0;0];
%         end
%     else
%         if k*agents{1}.dt>t_human_withdraw
%             v_man=[0;0];
%         end
%     end
%     last_human_pos = x_true(3:4);
%     x_true(1:2) = x_next_no_spec_human_motion(1:2);
%     x_true(3:4) = last_human_pos + agents{1}.dt*v_man + agents{1}.dt*u_for_true(3:4);
%     
        % Get observation model jacobians
%     z = agents{1}.obsModel.getObservation(x_true,'truenoise'); % true observation
    %truely observed output is not the one modeled by ekf
%     speed_man = norm(v_man);
%     direction_man = atan2(v_man(2),v_man(1));
%     %very problematic when human gives no more output
%     z(1:2)=[speed_man;direction_man] + chol(agents{1}.obsModel.R_speed)' * randn(2,1);
%     
    %% now do the machine part
    for i = 1:4%agents_amount
        % b_next already contains all info of mu_i and sig_i
        [b_next_i,mu_i,sig_i,~] = agents{i}.getNextEstimation(b{i},u{i},z{i});
        %should pass in the control and measurement of connected agents in
        %order to do estimations for them!
        b{1}(i,:)=b_next_i;
        b{2}(i,:)=b_next_i;
        b{3}(i,:)=b_next_i;
        b{4}(i,:)=b_next_i;
        for i_comp=1:agents{i}.components_amount % only one component
            mu{i,i_comp}=mu_i{i_comp};
            sig{i,i_comp}=sig_i{i_comp};
%             weight{i}(i_comp)=weight_i(i_comp);
        end
    end
%     [b_next_i,mu_i,sig_i,weight_i] = agents{1}.getNextEstimation(b{1},u,z);

    %% now for save
    for i = 1:4%agents_amount
        for i_comp = 1 : agents{i}.components_amount
            mu_save{i,i_comp}(:,k+1) = mu{i,i_comp};
            sig_save{i,i_comp}(:,k+1) = sig{i,i_comp}(:);
%             weight_save{i,i_comp}(:,k+1) = weight{i}(i_comp);
        end
    end
    x_save(:,:,k+1) = x_true;
    x_true_final = x_true;

    
    if k>1
        figure(6)

        plot(mu_save{1,1}(1,k),mu_save{1,1}(2,k),'bo')
        hold on
        grid on
        axis equal
        plot(mu_save{2,1}(1,k),mu_save{2,1}(2,k),'ro')
        plot(mu_save{3,1}(1,k),mu_save{3,1}(2,k),'ko')
        plot(mu_save{4,1}(1,k),mu_save{4,1}(2,k),'go')
        pointsToPlot = drawResult([mu_save{1}(:,k); sig_save{1}(:,k)], 2);
        plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
        pointsToPlot = drawResult([mu_save{2}(:,k); sig_save{2}(:,k)], 2);
        plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
        pointsToPlot = drawResult([mu_save{3}(:,k); sig_save{3}(:,k)], 2);
        plot(pointsToPlot(1,:),pointsToPlot(2,:),'k')
        pointsToPlot = drawResult([mu_save{4}(:,k); sig_save{4}(:,k)], 2);
        plot(pointsToPlot(1,:),pointsToPlot(2,:),'g')
        plot(z{1}(1),z{1}(2),'b*')
        plot(z{2}(1),z{2}(2),'r*')
        plot(z{3}(1),z{3}(2),'k*')
        plot(z{4}(1),z{4}(2),'g*')
        plot([x_save(1,1,k-1),x_save(1,1,k)],[x_save(1,2,k-1),x_save(1,2,k)],'-r.')
        plot([x_save(2,1,k-1),x_save(2,1,k)],[x_save(2,2,k-1),x_save(2,2,k)],'-k.')
        plot([x_save(3,1,k-1),x_save(3,1,k)],[x_save(3,2,k-1),x_save(3,2,k)],'-r.')
        plot([x_save(4,1,k-1),x_save(4,1,k)],[x_save(4,2,k-1),x_save(4,2,k)],'-r.')
    %     
    %     pointsToPlot = drawResultGMM([mu_save{1,1}(:,k); sig_save{1,1}(:,k)], agents{1}.motionModel.stDim);
    %     plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
    %     pointsToPlot = drawResultGMM([mu_save{1,2}(:,k); sig_save{1,2}(:,k)], agents{1}.motionModel.stDim);
    %     plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')

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
        pause(0.2);
    end
end
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
end