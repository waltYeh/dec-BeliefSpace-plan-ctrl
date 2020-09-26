% R_assists_t = diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])/3;
% Qerr_t = 0.0*eye(2);
% Qerr_l = 10*L*eye(2); % penalize terminal error
% Q_formation = 0*eye(2);

mm = HumanMind(dt); % motion model

om = HumanReactionModel(); % observation model

horizonSteps=51;
DYNCST  = @(b,u,i) beliefDynCost_assisting_robot_centralized(b,u,horizonSteps,false,mm,om);
[x_centr,cost_new_centr] = iLQG_rollout_centr(DYNCST, b0_centr, u_opt);
sum(cost_new_centr)
[x_admm,cost_new_admm] = iLQG_rollout2(interfDiGr,1,agents{1}.dyn_cst, b0_admm(1,:), agents{1}.u_nom);
sum(cost_new_admm)
plot(squeeze(cost_new_admm))
hold on
plot(squeeze(cost_new_centr))