addpath(genpath('./'));
options.rho=10;
options.gam=1;
options.reweightedIter=5;
show_mode=2;
[G,W]=sparse_feedback(L_opt,diffs,b_nom,options);
mm = HumanMind(dt); % motion model
om = HumanReactionModel(); % observation model
[didCollide, b0_next, x_true_final] = mpc_centralized_animateGMM(18,19,b0, b_nom, ...
    u_nom, G, update_steps,time_past, mm, om,Op.lims, show_mode);
for k=1:size(G,3)
    if sum(G(5:10,1:20,k),'all')~=0
        k
        a=1111
    end
    if sum(G(5:10,22:41,k),'all')~=0
        k
        b=1111
    end
end