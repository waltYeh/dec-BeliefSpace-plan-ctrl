options.rho=1;
options.gam=0.1;
options.reweightedIter=5;
show_mode=2;
[G,W]=sparse_feedback(L_opt,diffs,b_nom,options);
G(5,21,25)
mm = HumanMind(dt); % motion model
om = HumanReactionModel(); % observation model
[didCollide, b0_next, x_true_final] = mpc_centralized_animateGMM(35,36,b0, b_nom, ...
    u_nom, G, update_steps,time_past, mm, om,Op.lims, show_mode);