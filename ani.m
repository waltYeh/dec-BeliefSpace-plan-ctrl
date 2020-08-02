addpath(genpath('./'));
% clear
% close all
t0 = 0;
dt = 0.05;
tf = 2.0;
tspan = t0 : dt : tf;
nSteps = length(tspan);

motionModel = HumanMind(dt); % motion model

obsModel = HumanReactionModel(); % observation model
agent=AgentPlattform(dt,nSteps);

mu_1 = [8.5, 4.0, 5.0, 0.0]';
mu_2 = [3, 2.0, 5.0, 0.0]';
sig_1 = diag([0.01, 0.01, 0.01, 0.01]);%sigma
sig_2 = diag([0.01, 0.01, 0.01, 0.01]);
weight_1 = 0.6;
weight_2 = 0.4;
b0=[mu_1;sig_1(:);weight_1;mu_2;sig_2(:);weight_2];
Op.lims  = [-0.0 0.0;
    -4.0 4.0;
    -0.0  0.0;
    -0.0  0.0;
    -2.0 2.0;
    -2.0 2.0];
[failed, b_f] = animateGMM(7,8,b0, [],[],[], nSteps, motionModel, obsModel,Op.lims,10);