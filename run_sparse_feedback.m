addpath(genpath('./'));
options.rho=10;
options.gam=1;
options.reweightedIter=5;
% show_mode=3;
[G,W]=sparse_feedback(L_opt,diffs,b_nom,options);
mm = HumanMind(dt); % motion model
om = HumanReactionModel(); % observation model
[didCollide, b0_next, x_true_final] = mpc_centralized_animateGMM(101,102,b0_centr, b_nom, ...
    u_nom, L_opt, update_steps,time_past, mm, om,Op.lims, show_mode);
sparsity_pattern=ones(size(W));
for i=1:size(W,1)
    for j=1:size(W,2)
        if W(i,j)>50
            sparsity_pattern(i,j)=0;
        end
    end
end
figure(99)
spy(sparsity_pattern)
hold on
plot([0,67],[4.5,4.5],'k')
plot([0,67],[6.5,6.5],'k')
plot([0,67],[8.5,8.5],'k')
plot([0,67],[10.5,10.5],'k')
plot([42.5,42.5],[0,13],'k')
plot([48.5,48.5],[0,13],'k')
plot([54.5,54.5],[0,13],'k')
plot([60.5,60.5],[0,13],'k')