% R_assists_t = diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])/3;
% Qerr_t = 0.0*eye(2);
% Qerr_l 5394= 10*L*eye(2); % penalize terminal error
% Q_formation = 0*eye(2);

mm = HumanMind(dt); % motion model

om = HumanReactionModel(); % observation model

horizonSteps=51;
DYNCST  = @(b,u,i) beliefDynCost_assisting_robot_centralized(b,u,horizonSteps,false,mm,om);
[x_centr,cost_new_centr] = iLQG_rollout_centr(DYNCST, b0_centr, u_nom);
cost_new_centr=transpose(squeeze(cost_new_centr));
sum_centr=sum(cost_new_centr)

[x_admm,cost_new_admm_plat] = iLQG_rollout2(interfDiGr,1,agents{1}.dyn_cst, b0_admm(1,:), agents{1}.u_nom);
[x_admm,cost_new_admm_assist2] = iLQG_rollout2(interfDiGr,2,agents{2}.dyn_cst, b0_admm(2,:), agents{2}.u_nom);
[x_admm,cost_new_admm_assist3] = iLQG_rollout2(interfDiGr,3,agents{3}.dyn_cst, b0_admm(3,:), agents{3}.u_nom);
[x_admm,cost_new_admm_assist4] = iLQG_rollout2(interfDiGr,4,agents{4}.dyn_cst, b0_admm(4,:), agents{4}.u_nom);
[x_admm,cost_new_admm_assist5] = iLQG_rollout2(interfDiGr,5,agents{5}.dyn_cst, b0_admm(5,:), agents{5}.u_nom);
cost_new_admm_plat=squeeze(cost_new_admm_plat);
cost_new_admm_assist2=squeeze(cost_new_admm_assist2);
cost_new_admm_assist3=squeeze(cost_new_admm_assist3);
cost_new_admm_assist4=squeeze(cost_new_admm_assist4);
cost_new_admm_assist5=squeeze(cost_new_admm_assist5);
cost_new_admm=cost_new_admm_plat+cost_new_admm_assist2+cost_new_admm_assist3+cost_new_admm_assist4+cost_new_admm_assist5;
sum_admm=sum(cost_new_admm)
figure(58)
plot(squeeze(cost_new_admm))
hold on
plot(squeeze(cost_new_centr))
legend('admm','centr')